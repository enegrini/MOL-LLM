import os
import io
import sys
import copy
import json
from collections import defaultdict
import time
import h5py

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from .dict_maker import dictionary_maker
from itertools import chain


def get_file_handler(path):
    # export_path_prefix = os.path.join(path, "data.prefix")  # need to change this if saving in multiple files
    assert path.endswith(".prefix")
    export_path_prefix = path
    file_handler_prefix = io.open(export_path_prefix, mode="a", encoding="utf-8")
    print(f"Data will be stored in prefix in: {export_path_prefix} ...")
    return file_handler_prefix


def export_data(file_handler_prefix, samples, save_data_matrix, save_control_matrix, save_coeff_matrix, cur_line):
    """
    Export data to the disk.
    INPUTS:
        file_handler_prefix: file handler for the dataset saving location (.prefix file)
        samples: a dictionary where values are lists containing value for each sample in a batch
    """

    for i in range(len(samples["data"])):  # added range()
        outputs = dict()
        outputs["type"] = samples["type"][i]

        # data = samples["data"][i]
        # data = data.tolist()  # save data as list of floats
        # outputs["data"] = data
        # print('len samples[data], cur_line, i',len(samples["data"]),cur_line+i,i)
        save_data_matrix[cur_line+i] = samples["data"][i].numpy()

        # control = samples["control"][i]
        # control = control.tolist()  # save data as list of floats
        # outputs["control"] = control
        save_control_matrix[cur_line+i] = samples["control"][i].numpy()
        save_coeff_matrix[cur_line+i] = samples["coefficients"][i].numpy()

        outputs["text"] = samples["text"][i]

        file_handler_prefix.write(json.dumps(outputs) + "\n")
        file_handler_prefix.flush()


def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico


class MyDataset(Dataset):
    def __init__(
        self,
        train,
        config,
        FLAGS,
        size=None,
    ):
        super(MyDataset).__init__()
        self.train = train
        self.env_base_seed = config["env_base_seed"]
        self.max_dim = config["max_dimension"] #max data dimension (3D for Lorenz) (128D for PDE/space dimension)
        self.max_number_coeffs = config["max_number_coeffs"]
        self.text_path = (
            config["folder"] + config["text_filename"]
        )  # string for dataset location, change later if loading from multiple files
        self.data_path = config["folder"] + config["data_filename"]
        self.export_data = FLAGS.export_data
        self.t_len = FLAGS.t_len
        self.t_start = FLAGS.t_start
        self.t_end = FLAGS.t_end
        self.nIC_per_eq = FLAGS.IC_per_eq
        self.IC_types = FLAGS.IC_types
        self.count = 0
        self.remaining_data = 0
        self.config = config
        self.FLAGS = FLAGS
        # self.errors = defaultdict(int)

        if "test_env_seed" in config:
            self.test_env_seed = config["test_env_seed"]
        else:
            self.test_env_seed = None

        # self.batch_load = config["batch_load"]
        # self.reload_size = config["reload_size"]
        # self.local_rank = FLAGS.local_rank

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if not self.export_data:
            assert os.path.isfile(self.text_path), f"{self.text_path} not found"
            with io.open(self.text_path, mode="r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    # if i % FLAGS.n_gpu_per_node == FLAGS.local_rank:
                    lines.append(json.loads(line.rstrip()))

                self.data = lines

                print(f"Loaded {len(self.data)} equations from the disk.")
            self.size = len(self.data)

            assert os.path.isfile(self.data_path), "Data file {} not found".format(self.data_path)
            with h5py.File(self.data_path, "r") as hf:
                self.data_matrix = hf["data"][:] #(n_eqs,times,dim+1) for PDES (n_eqs,times,spaces, dim+1) or better (n_eqs,times x spaces, dim+1)
                self.control_matrix = hf["control"][:]
                self.coeff_matrix = hf["coefficients"][:]

            print('data-matrix len',len(self.data_matrix))
            print('data len',len(self.data))
            assert len(self.data_matrix) == len(self.data), "Dataset size mismatch"
            assert len(self.data_matrix) == len(self.control_matrix), "Dataset size mismatch"

            print(f"Data size: {self.data_matrix.shape}")
        else:
            self.size = size if size is not None else 10000000

    # def load_chunk(self):
    #     self.basepos = self.nextpos
    #     # print(f"Loading data from {self.text_path} ... seekpos {self.seekpos}, " f"basepos {self.basepos}")
    #     endfile = False
    #     with io.open(self.text_path, mode="r", encoding="utf-8") as f:
    #         f.seek(self.seekpos, 0)
    #         lines = []
    #         for i in range(self.reload_size):
    #             line = f.readline()
    #             if not line:
    #                 endfile = True
    #                 break
    #             if i % self.FLAGS.n_gpu_per_node == self.local_rank:
    #                 lines.append(line.rstrip().split("|"))
    #         self.seekpos = 0 if endfile else f.tell()

    #     self.data = [xy.split("\t") for _, xy in lines]
    #     self.data = [xy for xy in self.data if len(xy) == 2]
    #     self.nextpos = self.basepos + len(self.data)
    #     # logger.info(
    #     #     f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
    #     #     f"nextpos {self.nextpos}"
    #     # )
    #     if len(self.data) == 0:
    #         self.load_chunk()

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        samples = zip_dic(elements)
        # errors = copy.deepcopy(self.errors)
        # self.errors = defaultdict(int)
        
        return samples

    #     def init_rng(self):
    #         """
    #         Initialize random generator for training.
    #         """
    #         if self.rng is not None:
    #             return
    #         if self.train:
    #             worker_id = self.get_worker_id()
    #             self.worker_id = worker_id
    #             seed = [worker_id, self.global_rank, self.env_base_seed]
    #             self.rng = np.random.RandomState(seed)
    #             logger.info(
    #                 f"Initialized random generator for worker {worker_id}, with seed "
    #                 f"{seed} "
    #                 f"(base seed={self.env_base_seed})."
    #             )
    #         else:
    #             worker_id = self.get_worker_id()
    #             self.worker_id = worker_id
    #             seed = [
    #                 worker_id,
    #                 self.global_rank,
    #                 self.test_env_seed,
    #             ]
    #             self.rng = np.random.RandomState(seed)
    #             logger.info(
    #                 "Initialized {} generator, with seed {} (random state: {})".format(
    #                     self.type, seed, self.rng
    #                 )
    #             )

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0), "issue in worker id"
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        #         self.init_rng()
        if self.export_data:
            sample = self.generate_sample(index)
        else:
            sample = self.read_sample(index)

        return sample

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        # if self.train and self.batch_load:
        #     if index >= self.nextpos:
        #         self.load_chunk()
        #     idx = index - self.basepos

        x = dict()
        x["type"] = self.data[idx]["type"]
        if x["type"]<6:
            x['dim'] = 1
        elif (x["type"]==6 or x["type"]==7):
            x['dim'] = 3
        elif x["type"]>7 and x["type"]<13:
            x['dim'] = 2
        elif x["type"]>12:
            x['dim'] = 128

        x["text"] = copy.deepcopy(self.data[idx]["text"])

        func_value = torch.from_numpy(self.data_matrix[idx]).float() #(times,dimension+1) (only 1 ic/parameter selected)
        #for PDE would be (times,spaces,dimension+1) or better (times x spaces, dim+1)
           
        x["data"] = func_value[0:1]  # (1,dimension+1) select only first time, dim + 1 because first entry is time
        #for PDE: if input  is (times x spaces, dim+1) then 
        #x["data"] = func_value[0:spaces] would be (spaces,dimension+1)

        control = torch.from_numpy(self.control_matrix[idx]).float() #(times,dimension+1)
        x["control"] = control

        coefficients = torch.from_numpy(self.coeff_matrix[idx]).float() #(times,dimension+1)
        x["coefficients"] = coefficients

        x["label"] = func_value[1:, 1:]  #(times-1,dimension) future solution values
        #for PDE: if input  is (times x spaces, dim+1) then 
        #x["label"] = func_value[spaces:, 1:] would be (times-1 x spaces,dimension)

        return x

    def gen_expr(self):
        # TODO: if want to use larger num_workers to speed up dataset generation,
        # need to call init_rng() in each subprocess and use self.rng to do random sampling
        # (this will prevent different process generating the same data)
        sentence_ids = [int(x) for x in self.FLAGS.sentence_ids]
        data = dictionary_maker(self.max_dim, self.max_number_coeffs, sentence_ids,self.t_start, self.t_end, self.t_len,self.nIC_per_eq, self.IC_types) # sentence_ids are the id of the equation we want to use for data
        if sum(map(len, data)) == 0:
            assert False, "Error in data generation"
        for i,eq_idx in enumerate(sentence_ids): #(only selected indices)     for eq_idx in range(len(data)):
            for d in data[i]:
                d["type"] = eq_idx+1
                d["data"] = torch.from_numpy(d["data"]).float() #(n_eqs,times, dim+1) for PDES: (n_eqs,times,spaces, dim+1) or better (n_eqs,times x spaces, dim+1)
                d["control"] = torch.from_numpy(d["control"]).float()
                d["coefficients"] = torch.from_numpy(d["coefficients"]).float()

        return list(chain(*data)) ##list containing all the dictionaries from all equation kinds

    def generate_sample(self, index=None):
        """
        Generate a sample.
        """
        if self.remaining_data == 0:
            self.item = self.gen_expr()
            self.remaining_data = len(self.item)
            assert self.remaining_data > 0, "Not generating new data"

        self.remaining_data -= 1
        sample = self.item[-self.remaining_data]

        self.count += 1
        return sample


class InfiniteDataLooper:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.data_iter_num = 0

    def __next__(self):
        try:
            out = next(self.data_iter)
        except StopIteration:
            print(f"reached end of data loader, restart {self.data_iter_num}")
            self.data_iter_num += 1
            self.data_iter = iter(self.data_loader)
            out = next(self.data_iter)
        return out
