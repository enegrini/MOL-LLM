import torch
import torch.nn as nn


def batch_tokenize(tokenizer, texts):
    """
    Tokenize texts within a batch
    INPUTS:
        tokenizer: e.g. GPT2Tokenizer.from_pretrained("gpt2")
        texts:     nested list of dimension (B, N) where B is batch size, N is number of split strings
                   in one sentence. Each element is a string to be tokenized.
    OUTPUT:
        tokenized texts of shape (B, N, slen). No special tokens (bos or eos or pad) are added.
    """

    res = [[tokenizer.encode(sub_str[0], add_special_tokens=False) for i, sub_str in enumerate(seq)] for seq in texts]
    # changed sub_str into sub_str[0] in the .encode part since sub_str=['text'] so we need to extraxt 'text' before ecoding

    return res

def number_tokenize(tokenizer, numeric_data, dimensions):
    """INPUTS:
        tokenizer: GPT model tokenizer to find text vocabulary dimension. Assumes "<number_token>" and "<pad>" tokens are already added to vocab.
        data: nested list of dimension (N, B, data_len, data_dimension) where B is batch size, N is number of data blocks (may be different than number of text blocks)
        for now N = 3 as it has data, control, coefficients
        Each element is a numeric data to be "tokenized" (to be assigned dummy token)
    OUTPUT:
        List[Tensor] (N_out, B, data_len=timesteps) constant tensor with dummy number token.
        for now N_out = 2 as it has data, control/coefficients (depending on dimension)
    Note: data_len=timesteps is assumed to be the same for control and solution for now or we need to use nested lists
    """

    # Check the added token in the vocabulary
    number_token_index = tokenizer.convert_tokens_to_ids("<number_token>")  # number_token index

    # Determine the dimensions of the numeric data
    N = len(numeric_data)
    B = len(numeric_data[0])
    tokenized_data = []
    #first entry of numeric_data is always "data" so time dimension is the same for 1D and 3D
    data_len = len(numeric_data[0][0])
    # Create a tensor filled with the dummy number token
    constant_tensor = torch.full((B, data_len), number_token_index, dtype=torch.long)
    tokenized_data.append(constant_tensor)

    temp = []
    for j in range(B):
        #################uncomment if there is control in 1D eq#################################
        # if dimensions[j]==1:
        #     data_len = len(numeric_data[1][j]) #len_time for controls
        #     # Create a tensor filled with the dummy number token
        #     constant_tensor = torch.full((data_len,), number_token_index, dtype=torch.long)
        #     temp.append(constant_tensor)
        # else:
        data_len = 1  #coefficients are constant in time
        # Create a tensor filled with the dummy number token
        constant_tensor = torch.full((data_len,), number_token_index, dtype=torch.long)
        temp.append(constant_tensor)
    tokenized_data.append(temp)
    return tokenized_data

# def number_tokenize(tokenizer, numeric_data):
#     """INPUTS:
#         tokenizer: GPT model tokenizer to find text vocabulary dimension. Assumes "<number_token>" and "<pad>" tokens are already added to vocab.
#         data: nested list of dimension (N, B, data_len, data_dimension) where B is batch size, N is number of data blocks (may be different than number of text blocks)
#               Each element is a numeric data to be "tokenized" (to be assigned dummy token)
#     OUTPUT:
#         List[Tensor] (N, B, data_len=timesteps) constant tensor with dummy number token.
#     Note: data_len=timesteps is assumed to be the same for control and solution for now or we need to use nested lists
#     """

#     # Check the added token in the vocabulary
#     number_token_index = tokenizer.convert_tokens_to_ids("<number_token>")  # number_token index

#     # Determine the dimensions of the numeric data
#     N = len(numeric_data)
#     B = len(numeric_data[0])
#     tokenized_data = []
#     for i in range(N): #data lenght changes with N
#         data_len = len(numeric_data[i][0])

#         # Create a tensor filled with the dummy number token
#         constant_tensor = torch.full((B, data_len), number_token_index, dtype=torch.long)
#         tokenized_data.append(constant_tensor)
#     return tokenized_data


def get_word_embeddings(tokenized_texts, model):
    """
    Replace word ids with word embeddings
    INPUTS:
        tokenzied_texts: nested integer lists of size (B, N, slen)
        model:           e.g. GPT2Model.from_pretrained("gpt2")
    OUTPUT:
        nested lists of word embeddings (B, N, slen, hidden_dim), 2 inner most layers are Tensor.
    """
    res = [
        [model.wte(torch.LongTensor(sub_str).to(model.device).unsqueeze(0)).squeeze(0) for sub_str in seq]
        for seq in tokenized_texts
    ]
    return res
#TODO check this and change 3 to max_dim
def get_data_embeddings(data, dimensions, mlp2_data, mlp2_control, mlp2_coeffs):
    """
    mlp embeddings now based on data dimension. 
    
    INPUTS:
        data: nested list of dimension (N, B, slen, data_components) 
        for now N = 3 as it has data, control, coefficients
        dimensions: tensor of dimension (B,) specifying dimensionality for each batch element
        mlp2_data: MLP function for data[0]
        mlp2_control: MLP function for data[1]
        mlp2_coeffs: MLP function for data[2]
        
    OUTPUT:
        List[Tensor] (N_out, B, data_len=timesteps, d=768)
        for now N_out = 2 as it has data, control/coefficients (depending on dimension)
    """
    #for PDEs data[0] is : (B, times,spaces, data_components)

    temp = []
    res = []
    # Apply mlp2_data to data[0]
    res.append(mlp2_data(torch.stack(data[0]).cuda())) # will work the same for PDE input (B, times,spaces, data_components)
    #for PDEs data ouptut will be of dimension (B, times x spaces, d = 768)

    ############if 1D ODEs have cobntrol but not coefficients use the following##############################
    ## Apply mlp2_control to data[1] if dimensions is 1
    # mask_control = (torch.tensor(dimensions) == 1).unsqueeze(-1).unsqueeze(-1).float().cuda()
    # temp.append(mlp2_control(torch.stack(data[1]).cuda()) * mask_control)
    # # Apply mlp2_coeffs to data[2] if dimensions > 1
    # mask_coeffs = (torch.tensor(dimensions)>1).unsqueeze(-1).unsqueeze(-1).float().cuda()
    # temp.append(mlp2_coeffs(torch.stack(data[2]).unsqueeze(1).cuda()) * mask_coeffs)
    # # Combine res[1] and res[2] based on dimensions
    # #if dimension at batch i is 1 then keep contol otherwise keep coefficients
    # combined_res = []
    # for i in range(len(data[0])):
    #     if dimensions[i]==1:
    #         combined_res.append(temp[0][i])
    #     else:
    #         combined_res.append(temp[1][i])

    # res.append(combined_res)

    ##################if 1D ODE only have coefficients:################################
    # Apply mlp2_coeffs to data[2] if dimensions > 0
    mask_coeffs = (torch.tensor(dimensions)>0).unsqueeze(-1).unsqueeze(-1).float().cuda()
    temp.append(mlp2_coeffs(torch.stack(data[2]).unsqueeze(1).cuda()) * mask_coeffs)


    # Combine res[1] and res[2] based on dimensions
    #if dimension at batch i is 1 then keep contol otherwise keep coefficients
    combined_res = []
    for i in range(len(data[0])):
        combined_res.append(temp[0][i])

    res.append(combined_res)
    return res


# def get_data_embeddings(data, dimensions, mlp2_data, mlp2_control, mlp2_coeffs):
#     """
#     Replace data with mlp embeddings
#     INPUTS:
#         data: nested list of dimension (N, B, slen, data_components) where B is batch size, N is number of data blocks (may be different than number of text blocks), data_components for 1D is 2 (time,data)
#         mlp2: 2 layer MLP
#     OUTPUT:
#         List[Tensor] (N, B, data_len=timesteps, d=768)
#     Note: data_len=timesteps is assumed to be the same for control and solution for now or we need to use nested lists
#     """
#     res = []
#     print("dimensions", dimensions)
#     print("datashape", torch.stack(data[0]))
#     print("control shape", torch.stack(data[1]))
#     res.append(mlp2_data(torch.stack(data[0]).cuda()))
#     res.append(mlp2_control(torch.stack(data[1]).cuda()))

#     # N = len(data)
#     # res = []
#     # for i in range(N):
#     #     res.append(mlp2(torch.stack(data[i]).cuda()))
#     print(res[0].shape, res[1].shape)
#     return res


def batch_indices(data, texts, tokenizer):
    """
    Batch different parts of input indices together for text generation
    INPUTS:
        data_tokens:  List[Tensor] (N_data, B, data_len) N_data is number of data blocks (Ex: u(t),c(t) , then N_data =2)
        text_tokens: List[List[List]] (B, N, text_len) N is number of text blocks (Ex: with text description N=3, without N=2)
        tokenizer: used to find vocab_size to define a pad_token_id outside of the vocabulary
    OUTPUTS:
        input_index_tensor: (B, max_len+1) text and numeric data indices for text generation (with end of text sequence after text description to separate from padding)
    """
    B = len(texts)
    # print("B", B)
    N = len(texts[0])  # N may be different for text and data if there is description
    N_data = len(data)  # Number of data blocks we have (only u and control for now)

    lengths = []

    # Get the special token index for padding
    pad_index = tokenizer.convert_tokens_to_ids("<pad>")  # pad token index

    # find lenghts of each sentence+data
    if N != N_data:  # if we have text description at the end, we have one more text block than data
        # print("more text blocks than data blocks: text descriptions")
        # assumes we start with text
        for i in range(B):
            cur_len = len(texts[i][0])
            for n in range(N_data):
                cur_len += len(data[n][i])
                #for PDEs data[0][i] will have dimension (times x spaces, d)
                cur_len += len(texts[i][n + 1])
            lengths.append(cur_len)

        # find max lenght (pad with zeros shorter sentences)
        lengths = torch.LongTensor(lengths).to(device=data[0].device)
        max_len = lengths.max().item()
        # Now use pad_index to create a tensor filled with the pad token
        # dimension is max_len+1 because we add end of sequence index after text description
        input_index_tensor = torch.full((B, max_len + 1), pad_index, dtype=data[0].dtype, device=data[0].device)

        # re-organize embedding to have data and text in the correct position
        for i in range(B):
            cur_index = 0
            cur_text = texts[i][0]
            next_index = cur_index + len(cur_text)
            input_index_tensor[i, cur_index:next_index] = torch.tensor(cur_text)
            for n in range(N_data):
                cur_data = data[n][i]
                cur_index = next_index
                next_index = cur_index + cur_data.size(0)
                input_index_tensor[i, cur_index:next_index] = cur_data

                cur_text = texts[i][n + 1]
                cur_index = next_index
                next_index = cur_index + len(cur_text)
                input_index_tensor[i, cur_index:next_index] = torch.tensor(cur_text)

                cur_index = next_index
            # after input_index_tensor has been populated, add end of text index (only done if we have text description)
            input_index_tensor[i, cur_index] = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    else:  # if no text description we have same text and data blocks (we do not add end of text index in this case)
        print("same text blocks as data blocks: no descriptions")
        for i in range(B):
            cur_len = 0
            for n in range(N):
                cur_len += len(data[n][i])
                #for PDEs data[0][i] will have dimension (times,spaces, d) or maybe we can do (times x spaces, d)
                cur_len += len(texts[i][n])
            lengths.append(cur_len)

        # find max lentght (pad with zeros shorter sentences)
        lengths = torch.LongTensor(lengths).to(device=data[0].device)
        max_len = lengths.max().item()
        input_index_tensor = torch.full((B, max_len), pad_index, dtype=data[0].dtype, device=data[0].device)

        # re-organize embedding to have data and text in the correct position

        for i in range(B):
            cur_index = 0
            for n in range(N):
                cur_text = texts[i][n]
                next_index = cur_index + len(cur_text)
                input_index_tensor[i, cur_index:next_index] = torch.tensor(cur_text)

                cur_data = data[n][i]
                cur_index = next_index
                next_index = cur_index + cur_data.size(0)
                input_index_tensor[i, cur_index:next_index] = cur_data

                cur_index = next_index
    return input_index_tensor


def batch_inputs(data, texts):
    """
    Batch different parts of input together
    INPUTS:
        data_embedding:  List[Tensor] (N_data, B, data_len, d) N_data is number of data blocks (Ex: u(t),c(t) , then N_data =2)
        text_embedding: List[List[Tensor]] (B, N, text_len, d) N is number of text blocks (Ex: with text description N=3, without N=2)
    OUTPUTS:
        input_tensor: (B, max_len, d)
        lengths:      LongTensor(B, )
        input_lengths: LongTensor(B, ) lenghts of sequences without description for numeric output
        token_type:   LongTensor(B, max_len), 0 for text and 1 for data.
        position_encoding: LongTensor(B, max_len) position encoding changes for text but not for data
                           Example: token_type=[0,0,0,1,1] --> position_encoding=[1,2,3,4,4]
        text_mask: mask that selects only description (no input nor padding)
        input_mask: mask that selects only input (no description or padding)

    """
    B = len(texts)
    N = len(texts[0])  # N may be different for text and data if there is description
    N_data = len(data)  # Number of data blocks we have (only u and control/coefficients for now)

    lengths = []
    input_lengths = []

    # find lenghts of each sentence+data
    if N != N_data:  # if we have text description at the end, we have one more text block than data
        # assumes we start with text
        for i in range(B):
            cur_len = texts[i][0].size(0)
            for n in range(N_data):
                cur_len += len(data[n][i])#data[n].size(1)
                #for PDEs data[0][i] will have dimension (times x spaces, d)
                cur_len += texts[i][n + 1].size(0)
            lengths.append(cur_len)
            input_lengths.append(
                cur_len - texts[i][n + 1].size(0)
            )  # subtract lenght of text description to get input_lenght.

        # find max lentgh (pad with zeros shorter sentences)
        lengths = torch.LongTensor(lengths).to(device=data[0].device)
        input_lengths = torch.LongTensor(input_lengths).to(device=data[0].device)
        max_len = lengths.max().item()
        input_tensor = torch.zeros(B, max_len, data[0].size(-1), dtype=data[0].dtype, device=data[0].device)
        token_type = torch.zeros(B, max_len, dtype=torch.long, device=data[0].device)
        position_encoding = torch.zeros(B, max_len, dtype=torch.long, device=data[0].device)

        # re-organize embedding to have data and text in the correct position
        for i in range(B):
            cur_index = 0
            cur_text = texts[i][0]
            next_index = cur_index + cur_text.size(0)
            input_tensor[i, cur_index:next_index, :] = cur_text
            for n in range(N_data):
                cur_data = data[n][i]
                cur_index = next_index
                next_index = cur_index + cur_data.size(0)
                input_tensor[i, cur_index:next_index, :] = cur_data
                token_type[i, cur_index:next_index] = 1

                cur_text = texts[i][n + 1]
                cur_index = next_index
                next_index = cur_index + cur_text.size(0)
                input_tensor[i, cur_index:next_index, :] = cur_text

                cur_index = next_index
            current_position = 0
            for j in range(max_len):  # loop over token_type
                if token_type[i][j] == 0 or (j != 0 and token_type[i][j - 1] != token_type[i][j]):
                    current_position += 1  # increase position for text or when we encounter first data point.
                    # don't increase position until next text is detected
                position_encoding[i, j] = current_position

    else:  # if no text description we have same text and data blocks (and lenghts=input_lenghts)
        print("same text blocks as data blocks: no descriptions")
        for i in range(B):
            cur_len = 0
            for n in range(N):
                cur_len += len(data[n][i])#data[n].size(1)
                #for PDEs data[0][i] will have dimension (times x spaces, d)
                cur_len += texts[i][n].size(0)
            lengths.append(cur_len)

        # find max lentgh (pad with zeros shorter sentences)
        lengths = torch.LongTensor(lengths).to(device=data[0].device)
        input_lengths = lengths  # no text descirption so input_lenghts is same as lenghts
        max_len = lengths.max().item()
        input_tensor = torch.zeros(B, max_len, data[0].size(-1), dtype=data[0].dtype, device=data[0].device)
        token_type = torch.zeros(B, max_len, dtype=torch.long, device=data[0].device)
        position_encoding = torch.zeros(B, max_len, dtype=torch.long, device=data[0].device)

        # re-organize embedding to have data and text in the correct position

        for i in range(B):
            cur_index = 0
            for n in range(N):
                cur_text = texts[i][n]
                next_index = cur_index + cur_text.size(0)
                input_tensor[i, cur_index:next_index, :] = cur_text
                # token_type[i, cur_index:next_index] = 0

                cur_data = data[n][i]
                cur_index = next_index
                next_index = cur_index + cur_data.size(0)
                input_tensor[i, cur_index:next_index, :] = cur_data
                token_type[i, cur_index:next_index] = 1

                cur_index = next_index

            current_position = 0
            for j in range(max_len):  # loop over token_type
                if token_type[i][j] == 0 or (j != 0 and token_type[i][j - 1] != token_type[i][j]):
                    current_position += 1  # increase position for text or when we encounter first data point.
                    # don't increase position until next text is detected
                position_encoding[i, j] = current_position

    alength = torch.arange(max_len, dtype=torch.long, device=lengths.device)[None]  # (1, max_len)
    text_mask = torch.logical_and(
        alength < lengths[:, None], alength >= input_lengths[:, None] - 1
    )  # start from last element of input, entil before padding, so range is [input_len-1, length)
    input_mask = alength < input_lengths[:, None]-1
    
    return input_tensor, lengths, input_lengths, token_type, position_encoding, text_mask, input_mask
