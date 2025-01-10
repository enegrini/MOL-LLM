import torch
from absl import logging
import utils
import torch.optim as optim
import numpy as np
from tabulate import tabulate
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_gen.preprocess import (
    batch_tokenize,
    number_tokenize,
    get_word_embeddings,
    get_data_embeddings,
    batch_inputs,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, model_config, opt_config, trainable_mode, t_start =0.0, t_end = 3.0, t_len=300, amp=False):
        print("flash_sdp_enabled", torch.backends.cuda.flash_sdp_enabled())  # True
        self.model = model
        self.model_config = model_config
        self.opt_config = opt_config
        self.trainable_mode = trainable_mode
        self.max_dim = model_config["max_dimension"]
        print("trainable_mode: {}".format(self.trainable_mode), flush=True)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            print("model wrapped by DataParallel", flush=True)

        self.model.to(device)
        print("model moved to {}".format(device), flush=True)

        model = self.model.module if hasattr(self.model, "module") else self.model
        # if not (trainable_mode == "all"):  # freeze the model first
        #     for param in model.parameters():
        #         param.requires_grad = False

        # # Dictionary mapping the component to its parameter name pattern
        # patterns = {
        #     "unet": ["unet"],
        #     "transformer": ["transformer"],
        # }

        # for name, params in model.named_parameters():
        #     for mode, pattern_list in patterns.items():
        #         if any(pattern in name for pattern in pattern_list) and mode in trainable_mode:
        #             params.requires_grad = True

        headers = ["Parameter Name", "Shape", "Requires Grad"]
        table_data = [(name, str(param.shape), param.requires_grad) for name, param in model.named_parameters()]
        # print(tabulate(table_data, headers=headers, tablefmt="grid"))

        headers = ["Trainable Parameters", "Shape"]
        table_data = [(name, str(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
        # print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # TODO: If want to only finetune GPT2, maybe can set its learning rate to be smaller
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_config["peak_lr"],
            weight_decay=opt_config["weight_decay"],
        )
        self.lr_scheduler = utils.WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup=opt_config["warmup_steps"],
            max_iters=opt_config["decay_steps"],
        )
        self.amp = amp
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using automatic mixed precision", flush=True)
        # print(self.model, flush=True)
        self.train_step = 0

        # TODO: change queries based on label location
        self.queries = torch.linspace(start=t_start, end=t_end, steps=t_len, dtype=torch.float, device=device)[1:]
        

    def _loss_fn(self, samples, relative = True):

        with torch.cuda.amp.autocast(enabled=self.amp, dtype=torch.bfloat16):
            # (bs, query_len, output_dim), (batch_size, seq_length, vocab_size), (batch_size, seq_length+1), (batch_size, seq_length)
            numeric_output, text_output_logits, text_label_indices, text_mask = self.model(samples, self.queries)
            # print('TEXT LABEL DIM', text_label_indices.shape,text_mask.shape )
            label_numeric = torch.stack(samples["label"]).to(numeric_output.device)
            # Create a mask based on dim to mask out outputs for various dimensions 1D vs 3D
            bs = label_numeric.size(0)
            dims = torch.tensor(samples['dim']).to(numeric_output.device)
            # Create a range tensor and expand it
            mask_range = torch.arange(self.max_dim, device=numeric_output.device).expand(bs, 1, self.max_dim)
            # Create the mask by comparing with dimensions and then unsqueeze the dimensions tensor
            mask_dim = mask_range < dims.unsqueeze(1).unsqueeze(2)
            # Convert mask to float
            mask_dim = mask_dim.to(dtype=torch.float32)
            # Ensure mask_dim has the correct shape to be broadcast with numeric_output
            assert mask_dim.shape == (bs, 1, self.max_dim), f"mask_dim shape is incorrect: {mask_dim.shape}"
            loss_numeric = nn.MSELoss()(numeric_output * mask_dim, label_numeric * mask_dim)
            if relative:
                eps = 1e-7
                label_norm = torch.mean((label_numeric*mask_dim) ** 2)  
                loss_numeric = loss_numeric / (label_norm + eps) 
            
            text_mask = text_mask.to(text_output_logits.device)  # BoolTensor (batch_size, seq_length)
            label_text = text_label_indices[:, 1:].to(text_output_logits.device)[
                text_mask
            ]  # LongTensor (text_mask.sum(), )
            vocab_size = text_output_logits.size(-1)
            output_text = text_output_logits[text_mask.unsqueeze(-1).expand_as(text_output_logits)].view(
                -1, vocab_size
            )  # Tensor (text_mask.sum(), vocab_size)

            loss_text = nn.CrossEntropyLoss()(output_text, label_text)
        return loss_numeric, loss_text

    def iter(self, samples):
        """
        train the model, assume samples are directly from dataloader
        """
        self.model.train()
        loss_num, loss_text = self._loss_fn(samples) #defaults relative MSE
        loss = loss_num + loss_text  # how should we combine the two losses?
        self.optimizer.zero_grad()

        if not self.amp:  # regular training
            loss.backward()
            # Gradient clipping
            model = self.model.module if hasattr(self.model, "module") else self.model
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_config["gnorm_clip"])
            self.optimizer.step()
        else:  # using amp
            self.scaler.scale(loss).backward()
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            model = self.model.module if hasattr(self.model, "module") else self.model
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_config["gnorm_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.lr_scheduler.step()
        self.train_step += 1

    def save(self, save_dir, include_optimizer=True):
        model = self.model.module if hasattr(self.model, "module") else self.model

        data = {"model": model.state_dict()}
        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()

        torch.save(data, "{}/{}_params.pth".format(save_dir, self.train_step))
        logging.info("saved to {}, step {}".format(save_dir, self.train_step))

    def restore(self, save_dir, step, restore_opt_state=True):
        params_path = "{}/{}_params.pth".format(save_dir, step)
        model = self.model.module if hasattr(self.model, "module") else self.model
        data = torch.load(params_path, map_location=device)
        weights = data["model"]
        try:
            model.load_state_dict(weights)
        except RuntimeError:  # remove the 'module.'
            weights = {name.partition(".")[2]: v for name, v in weights.items()}
            model.load_state_dict(weights)

        logging.info("restored params from {}, step {}".format(save_dir, step))

        if restore_opt_state:
            self.optimizer.load_state_dict(data["optimizer"])
            logging.info("restored optimizer state from {}".format(save_dir))

    @torch.no_grad()
    def get_loss(self, samples):
        """
        assume raw data
        return numpy loss
        """
        self.model.eval()
        loss_num, loss_text = self._loss_fn(samples)
        # return loss.detach().cpu().numpy()
        print('computed numeric and text loss',loss_num, loss_text)
        return loss_num.item() + loss_text.item()

    @torch.no_grad()
    def get_pred(self, samples):
        """
        assume raw data
        return numpy predication of shape (bs, query_len, output_dim)
        """
        self.model.eval()
        with torch.cuda.amp.autocast(enabled=self.amp, dtype=torch.bfloat16):
            output = self.model.number_test_output(samples, self.queries)
        return output.float().numpy(force=True)

    def get_error(self, samples, relative=True):
        """
        assume raw data
        return numpy error
        """
        self.model.eval()
        output= self.get_pred(samples)  # (bs, query_len, output_dim)
        label = np.stack(samples["label"])
        bs = label.shape[0]
        # Create a numpy array of dimensions
        dims = np.array(samples['dim'])
        # Create a range array and expand it
        mask_range = np.arange(self.max_dim).reshape(1, 1, self.max_dim)
        mask_range = np.broadcast_to(mask_range, (bs, 1, self.max_dim))
        # Create the mask by comparing with dimensions
        dims = dims.reshape(bs, 1, 1)
        mask_dim = mask_range < dims
        # Convert mask to float
        mask_dim = mask_dim.astype(np.float32)
        # Ensure mask_dim has the correct shape to be broadcast with numeric_output
        assert mask_dim.shape == (bs, 1, self.max_dim), f"mask_dim shape is incorrect: {mask_dim.shape}"
        #compute masked error
        error = np.sqrt(np.mean((output*mask_dim - label*mask_dim) ** 2, axis=(1, 2)))  # (bs, )
        if relative:
            eps = 1e-7
            label_scale = np.sqrt(np.mean((label) ** 2, axis=(1, 2)))  # (bs, )
            error = error / (label_scale + eps)  # (bs, )

        error_mean = np.mean(error)
        error_std = np.std(error)
        return error_mean, error_std

    def get_plot_arrays(self, samples):
        output = self.get_pred(samples)  # (bs, query_len, output_dim)
        label = np.stack(samples["label"])
        queries_np = self.queries.detach().cpu().numpy()
        return output, label, queries_np
   
    ##TODO possibly add metric
    def test_text(self, samples, selected_indices = None, train = True, description=True):
        #assume batch size is 1 for testing
        self.model.eval()
        
        # get input tensor from sample
        tokenized_texts = batch_tokenize(self.model.model.tokenizer, samples["text"])
        word_embedding = get_word_embeddings(tokenized_texts, self.model.model.gpt2)
        data_embedding = get_data_embeddings([samples["data"], samples["control"],samples["coefficients"]],samples['dim'], self.model.embedder_data, self.model.embedder_control,self.model.embedder_coeffs)
        input_tensor, lengths, input_lengths, token_type_ids, position_ids, text_mask, input_mask = batch_inputs(
            data_embedding, word_embedding
        )
        #if the sample constains description, crop it out to get the input_tensor for testing
        if train: #only select first batch element
            if description:
                crop_index = input_lengths[0].item()
                input_tensor = input_tensor[0:1,:crop_index,:]
                token_type_ids = token_type_ids[0:1,:crop_index]
                position_ids = position_ids[0:1,:crop_index]
            # print('cropped input tensor shape',input_tensor.shape)
            #For the following we will use
            # model
            # input_tensor: the initial embedded input tensor of shape (batch_size, input_len, embedding_dim) (no descriptions)
            # model.wte: the embedding layer used to get the embeddings for text only(?) 
            # model.tokenizer: tokenizer that converts token indices to text
            # max_length: the maximum length of the generated sequence
        
            
            # Initialize a list to keep track of the generated token indices
            generated_sequence = []
            
            # Define the maximum number of tokens to generate
            max_length = 50

            # Get the end-of-text token ID from the tokenizer
            eot_token_id = self.model.model.tokenizer.eos_token_id
            
            # Start autoregressive generation
            for _ in range(max_length):
                # Step 1: Forward pass through the model to get logits
                logits = self.model.text_test_output(input_tensor, position_ids, token_type_ids)  # logits shape: (batch_size, seq_len, vocab_size)
            
                # Step 2: Get the probabilities for the next token
                probs = F.softmax(logits[:, -1, :], dim=-1)  # probs shape: (batch_size, vocab_size)
            
                # Step 3: Sample or select the next token
                next_token = torch.argmax(probs, dim=-1)  # next_token shape: (batch_size,)
            
                # Step 4: Append the next token to the generated sequence
                generated_sequence.append(next_token.item())  # assuming batch_size is 1
            
                # Step 5: Get the embedding of the next token
                next_token_embedding = self.model.model.gpt2.wte(next_token)  # next_token_embedding shape: (batch_size, embedding_dim)

                if next_token.item() == eot_token_id:
                    break
            
                # Step 6: Append the new token embedding to the input tensor and update position and token type ids
                input_tensor = torch.cat([input_tensor, next_token_embedding.unsqueeze(1)], dim=1)  # new input_tensor shape: (batch_size, seq_len + 1, embedding_dim)
                #add new position id (one more than last position id since we only add text)
                position_ids = torch.cat([position_ids, torch.tensor([[position_ids[:, -1].item()+1]],device=input_tensor.device)], dim=1) #(batch_size, seq_len + 1)
                #add new token type id 0 since we only add text
                token_type_ids = torch.cat([token_type_ids, torch.tensor([[0]],device=input_tensor.device)], dim=1) #(batch_size, seq_len + 1)
                
            # Convert the list of generated token indices to text
            generated_text = self.model.model.tokenizer.decode(generated_sequence)
            
            print("Generated Text: ", generated_text)
            return generated_text
        else:
            gen_text_list = []
            if selected_indices is not None:
                indices = selected_indices
            else:
                indices = list(range(len(input_lengths)))
            for i in indices:
                if description:
                    crop_index = input_lengths[i].item()
                    input_tensor_crp = input_tensor[i:i+1,:crop_index,:]
                    token_type_ids_crp = token_type_ids[i:i+1,:crop_index]
                    position_ids_crp = position_ids[i:i+1,:crop_index]
                #For the following we will use
                # model
                # input_tensor: the initial embedded input tensor of shape (batch_size, input_len, embedding_dim) (no descriptions)
                # model.wte: the embedding layer used to get the embeddings for text only(?) 
                # model.tokenizer: tokenizer that converts token indices to text
                # max_length: the maximum length of the generated sequence
            
                
                # Initialize a list to keep track of the generated token indices
                generated_sequence = []
                
                # Define the maximum number of tokens to generate
                max_length = 50

                # Get the end-of-text token ID from the tokenizer
                eot_token_id = self.model.model.tokenizer.eos_token_id
                
                # Start autoregressive generation
                for _ in range(max_length):
                    # Step 1: Forward pass through the model to get logits
                    logits = self.model.text_test_output(input_tensor_crp, position_ids_crp, token_type_ids_crp)  # logits shape: (batch_size, seq_len, vocab_size)
                
                    # Step 2: Get the probabilities for the next token
                    probs = F.softmax(logits[:, -1, :], dim=-1)  # probs shape: (batch_size, vocab_size)
                
                    # Step 3: Sample or select the next token
                    next_token = torch.argmax(probs, dim=-1)  # next_token shape: (batch_size,)
                
                    # Step 4: Append the next token to the generated sequence
                    generated_sequence.append(next_token.item())  # assuming batch_size is 1
                
                    # Step 5: Get the embedding of the next token
                    next_token_embedding = self.model.model.gpt2.wte(next_token)  # next_token_embedding shape: (batch_size, embedding_dim)

                    if next_token.item() == eot_token_id:
                        break
                
                    # Step 6: Append the new token embedding to the input tensor and update position and token type ids
                    input_tensor_crp = torch.cat([input_tensor_crp, next_token_embedding.unsqueeze(1)], dim=1)  # new input_tensor shape: (batch_size, seq_len + 1, embedding_dim)
                    #add new position id (one more than last position id since we only add text)
                    position_ids_crp = torch.cat([position_ids_crp, torch.tensor([[position_ids_crp[:, -1].item()+1]],device=input_tensor_crp.device)], dim=1) #(batch_size, seq_len + 1)
                    #add new token type id 0 since we only add text
                    token_type_ids_crp = torch.cat([token_type_ids_crp, torch.tensor([[0]],device=input_tensor_crp.device)], dim=1) #(batch_size, seq_len + 1)
                    
                # Convert the list of generated token indices to text
                generated_text = self.model.model.tokenizer.decode(generated_sequence)
                gen_text_list.append(generated_text)
                # print("Generated Text: ", generated_text)
            return gen_text_list
        
        