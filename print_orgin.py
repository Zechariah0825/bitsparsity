from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from torch.nn.parallel import DataParallel
import numpy as np
import functools
import struct
import os
import argparse
from argparse import ArgumentParser
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


# %%
#将group和path作为参数传入

if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    parser = ArgumentParser()
    parser.add_argument("--group", type=int, default=0, help="group number")
    parser.add_argument("--save_path", type=str, default="/data/gaozh/",help="path to save the result")
    parser.add_argument("--model_dir", type=str, default="/home/gaozh/Project/print_tensor/llama-7b",help="model path")
    parser.add_argument("--cuda_node", type=int, default=0,help="cuda node")
    parser.add_argument("--batch_size", type=int, default=4,help="batch size")
    parser.add_argument("--num_batches_to_iterate", type=int, default=0,help="num batches to iterate")
    parser.add_argument("--item", type=str, default="both",help="item type input or weight")
    parser.add_argument("--mode", type=bool, default=False,help="save orginal numpy or not")
    parser.add_argument("--speic_layer", type=str, default="all",help="speic layer name")


    args = parser.parse_args()

    group = args.group
    path = args.save_path
    model_dir = args.model_dir
    cuda_node = ("cuda:"+str(args.cuda_node))
    batch_size = args.batch_size
    num_batches_to_iterate = args.num_batches_to_iterate
    item = args.item
    mode = args.mode
    speic_layer=args.speic_layer

    print("group: ", group)
    print("save_path: ", path)
    print("model_dir: ", model_dir)
    print("cuda_node: ", cuda_node)
    print("batch_size: ", batch_size)
    print("num_batches_to_iterate: ", num_batches_to_iterate)
    print("item: ", item)
    print("mode: ", mode)
    print("speic_layer: ", speic_layer)

'''
group = 0
path = "/data/gaozh"
model_dir = "/home/gaozh/Project/print_tensor/llama-7b"
cuda_node = "cuda:0"
batch_size = 4
num_batches_to_iterate = 0
item = "both"
mode = True
speic_layer="down"
'''

# %%
device = torch.device(cuda_node)
datasets = load_dataset(path="wikitext", name="wikitext-103-v1", split="test")
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",load_in_8bit=True)

tokenizer.pad_token = tokenizer.eos_token
encoded_dataset = tokenizer(datasets['text'], padding=True, truncation=True, return_tensors="pt")

# %%
class MyDataset(Dataset):
    def __init__(self, input_tensor, attention_mask):
        self.input_tensor = input_tensor
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.attention_mask[idx]

# 创建数据集实例
dataset = MyDataset(encoded_dataset["input_ids"], encoded_dataset["attention_mask"])

# 定义批次大小
batch_size = 4

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%
print("Batch_num: ", len(dataloader))
print("Batch_size: ", batch_size)

# %%
folder_name=item+"_L"+speic_layer+"_G"+str(group)+"_M"+str(mode)+"_B"+str(batch_size)
folder_path = os.path.join(path, "print_tensor",folder_name)

# 检查路径是否存在文件夹，如果不存在则创建
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")
# %%

if not os.path.exists(folder_path+"/orgin"):
    os.makedirs(folder_path+"/orgin")
    print(f"Folder orign created successfully.")
else:
    print(f"Folder orign already exists.")
# %%

# %%

def float16_to_bin(num):
    # 将float16数打包为2字节16位，使用struct.pack
    packed_num = struct.pack('e', num)

    # 解包打包后的字节以获取整数表示
    int_value = struct.unpack('H', packed_num)[0]

    # 将整数表示转换为二进制
    binary_representation = bin(int_value)[2:].zfill(16)

    binary_np = np.array([int(i) for i in str(binary_representation)])
    return binary_np

# %%
#含有group的float16函数
def count_ones_in_binary_fp16(tensor,trans=False,group=0):
    # 获取张量的形状
    if trans:
        tensor = tensor.transpose(0, 1)

    num_rows, num_cols = tensor.shape
    
    # 初始化一个列表用于存储每一列的结果
    result_list = []

    if group == 0:
        for row in range(num_rows):
            row_data = tensor[row, :]  # 选择一行
            binary_np = [float16_to_bin(val) for val in row_data]
            result_list.append(binary_np)
        result_array = np.array(result_list)
        result_array=result_array.sum(axis=1)
        return result_array


    # 遍历每一列
    for row in range(num_rows):
        row_data = tensor[row,:]  # 选择一列

        # 初始化一个长度为16的列表用于存储每一位上1出现的次数
        
        for group_index in list(range(0,len(row_data),group)):
            star=group_index
            over=min(group_index+group,len(row_data)-1)
            if star<over:
                group_data = row_data[star:over]  # 选择一列中的一组
            elif star==over:
                group_data = [row_data[star]]
            else:
                print("error")
                break

            group_np = np.array([float16_to_bin(val) for val in group_data])
            group_ones=np.sum(group_np,axis=0)
            # 将每一列的结果添加到结果列表
            result_list.append(group_ones)

    # 转换为NumPy数组
    result_array = np.array(result_list)
    
    return result_array


# %%
test_tensor = np.array([[1, 2, 3],
                        [4, 5, 6]], dtype=np.int8)

# 调用函数进行测试，假设 trans 为 True，group 为 2
result = count_ones_in_binary_fp16(test_tensor, trans=False, group=0)

# 打印输出结果查看是否符合预期
print(result)

# %%
test_tensor = np.array([[1, 2, 3],
                        [4, 5, 6]], dtype=np.int8)

# 调用函数进行测试，假设 trans 为 True，group 为 2
result = count_ones_in_binary_fp16(test_tensor, trans=False, group=0)

# 打印输出结果查看是否符合预期
print(result)

# %%
def int8_to_bin(integers):
    binary_array = []
    for Num in integers:
        num=Num.item()
        if num >= 0:
            binary = '{:08b}'.format(num)
        else:
            # Convert negative numbers to their two's complement representation
            binary = '{:08b}'.format(256 + num)
        binary_array.append([bit for bit in binary[-8:]])  # Ensure only the last 8 bits are taken
    return np.array(binary_array, dtype=int)

# %%
#含有group的int8函数

def count_ones_in_binary_int8(tensor,trans=False,group=0):
    # 获取张量的形状
    if trans:
        tensor = tensor.transpose(0, 1)
        print(tensor.shape)

    num_rows, num_cols = tensor.shape
    
    # 初始化一个列表用于存储每一列的结果
    result_list = []

    if group == 0:
        for row in range(num_rows):
            row_data = tensor[row, :]
            binary_np = int8_to_bin(row_data)
            result_list.append(binary_np)
        result_array = np.array(result_list)
        result_array=result_array.sum(axis=1)
        return result_array

    # 遍历每一列
    for row in range(num_rows):
        row_data = tensor[row,:]  # 选择一列
        
        for group_index in range(0,len(row_data),group):
            star=group_index
            over=min(group_index+group,len(row_data)-1)
            if star<over:
                group_data = row_data[star:over]
            elif star==over:
                group_data = [row_data[star]]
            else:
                print("error")
                break
            group_np = int8_to_bin(group_data)
            group_ones = np.sum(group_np,axis=0)
            result_list.append(group_ones)

    # 转换为NumPy数组
    result_array = np.array(result_list)

    return result_array

# %%
#测试count_ones_in_binary_int8函数
test_tensor = np.array([[1, 2, 3],
                        [4, 5, 6]], dtype=np.int8)

test_tensor = torch.from_numpy(test_tensor)

a=count_ones_in_binary_int8(test_tensor,group=0)
print(a)

# %%
#测试count_ones_in_binary_int8函数
test_tensor = np.array([[1, 2, 3],
                        [4, 5, 6]], dtype=np.int8)

test_tensor = torch.from_numpy(test_tensor)

a=count_ones_in_binary_int8(test_tensor,trans=True,group=0)
print(a)

# %%
Batch_num=0

binary_input={}
binary_input_shape={}

binary_weight={}
binary_weight_shape={}

binary_output={}
binary_output_shape={}

orgin_input={}
orgin_weight={}
orgin_output={}


def print_layer_info(module, input, output,name,item,inter_path,folder_name,mean=False,orgin=False):

    global Batch_num

    global binary_input
    global binary_weight
    global binary_output

    global orgin_input
    global orgin_weight
    global orgin_output

    layer_name = module.__class__.__name__
    layer_info = str(module)
    print(f"module name: {name}")
    print(f"Layer name: {layer_name}")
    print(f"Layer info: {layer_info}")
    print(f"Layer dict: {module.state_dict().keys()}")
    
    if item=="both":
        if_input=True
        if_weight=True
        if_output=True
    elif item=="input":
        if_input=True
        if_weight=False
        if_output=False
    elif item=="weight":
        if_input=False
        if_weight=True
        if_output=False
    elif item=="output":
        if_input=False
        if_weight=False
        if_output=True


    if if_input:
        if isinstance(input, tuple):
            if len(input) > 0 and isinstance(input[0], torch.Tensor):
                print(f"Input shape: {input[0].shape}")
                row_dim = input[0].shape[-2]
                col_dim=input[0].shape[-1]
                x=input[0].view(-1, col_dim).detach()
                print(f"X type: {x.dtype}")
                print(f"X shape: {x.shape}")

                #如果orgin为True，不进行二进制转换，并且直接保存独立字典
                if orgin:
                    if Batch_num==1:
                        orgin_input[name]={}
                        orgin_input[name]={
                            'data':x,
                            'shape':[row_dim, col_dim]
                        }
                    else:
                        if orgin_input[name]['data'].shape[0]!=x.shape[0]:
                            return
                        else:
                            orgin_input[name]['data']=(orgin_input[name]['data']*(Batch_num-1)+x)/(Batch_num)
                else:

                    if mean:
                        byte=count_ones_in_binary_fp16(x,False,group).mean(axis=0)
                    else:
                        byte=count_ones_in_binary_fp16(x,False,group)
                    print(f"X byte shape: {byte.shape}")

                    if Batch_num==1:
                        binary_input[name]={}
                        binary_input[name]={
                            'byte':byte,
                            'shape':[row_dim, col_dim]
                        }

                    else:
                        if binary_input[name]['byte'].shape[0]!=byte.shape[0]:
                            return
                        else:
                            binary_input[name]['byte']=(binary_input[name]['byte']*(Batch_num-1)+byte)/(Batch_num)

                        
                    print(binary_input[name]['byte'])    
                    print(binary_input[name]['byte'].shape)

    
    if if_output:
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                print(f"Output shape: {output.shape}")
                row_dim = output.shape[0]*output.shape[1]
                col_dim=output.shape[-1]
                z=output.view(-1, col_dim).detach()
                print(f"Z shape: {z.shape}")
                print(f"Z type: {z.dtype}")

                if orgin:
                    if Batch_num==1:
                        orgin_output[name]={}
                        orgin_output[name]={
                            'data':z,
                            'shape':[row_dim, col_dim]
                        }
                    else:
                        if orgin_output[name]['data'].shape[0]!=z.shape[0]:
                            return
                        else:                        
                            orgin_output[name]['data']=(orgin_output[name]['data']*(Batch_num-1)+z)/(Batch_num)
                else:

                    if mean:
                        byte=count_ones_in_binary_fp16(z,False,group).mean(axis=0)
                    else:
                        byte=count_ones_in_binary_fp16(z,False,group)
                    print(f"Z byte shape: {byte.shape}")

                    if Batch_num==1:
                        binary_output[name]={}
                        binary_output[name]={
                            'byte':byte,
                            'shape':[row_dim, col_dim]
                        }

                    else:
                        if binary_output[name]['byte'].shape[0]!=byte.shape[0]:
                            return
                        else:
                            binary_output[name]['byte']=(binary_output[name]['byte']*(Batch_num-1)+byte)/(Batch_num)

                    print(binary_output[name]['byte'])    
                    print(binary_output[name]['byte'].shape)

    if if_weight:
        if Batch_num>1:
            return
        if 'weight' in module.state_dict().keys():
            weight=module.state_dict()['weight']
            print(f"Weight shape: {weight.shape}")
            row_dim = weight.shape[-2]
            col_dim=weight.shape[-1]
            y=weight.view(row_dim, -1).detach()
            print(f"Y shape: {y.shape}")
            print(f"Y type: {y.dtype}")

            if orgin:
                if Batch_num==1:
                    orgin_weight[name]={}
                    orgin_weight[name]={
                        'data':y,
                        'shape':[row_dim, col_dim]
                    }
                else:
                    if orgin_weight[name]['data'].shape[0]!=y.shape[0]:
                        return
            else:
                if Batch_num==1:
                    if mean:
                        byte=count_ones_in_binary_int8(y,True,group).mean(axis=0)
                    else:
                        byte=count_ones_in_binary_int8(y,True,group)
                    print(f"Y byte shape: {byte.shape}")
                    binary_weight[name]={}
                    binary_weight[name]={
                        'byte':byte,
                        'shape':[row_dim, col_dim]
                    }
                else:
                    if binary_weight[name]['byte'].shape[0]!=byte.shape[0]:
                        return
        
                print(binary_weight[name]['byte'])
                print(binary_weight[name]['byte'].shape)
                
# %%
np.set_printoptions(threshold=np.inf)



# %%

# %%
nums = [x for x in range(0, 31,5)]
print(f"target_layers:{nums}")

def format_layer_names(n):

    all_layer_names = [
        "model.layers.{}.self_attn.q_proj",
        "model.layers.{}.self_attn.k_proj",
        "model.layers.{}.self_attn.v_proj",
        "model.layers.{}.self_attn.o_proj",
        "model.layers.{}.mlp.gate_proj",
        "model.layers.{}.mlp.up_proj",
        "model.layers.{}.mlp.down_proj"]
    
    layer_names = []

    if speic_layer=="all":
        layer_names = all_layer_names

    else:
         for name in all_layer_names:
            if speic_layer in name:
                layer_names.append(name)

    formatted_layer_names = [name.format(n) for name in layer_names]
    return formatted_layer_names

def count_layer_names(nums):
    layer_names=[]
    for n in nums:
        layer_names=layer_names+format_layer_names(n)
    return layer_names


print(f"count_layer_names: {count_layer_names(nums)}")

# %%
# %%
hook_handles = []  # 用于存储钩子句柄
import functools
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Linear) and name in count_layer_names(nums):
        hook = layer.register_forward_hook(functools.partial(print_layer_info,name=name,mean=False,orgin=mode,inter_path=path,folder_name=folder_name,item=item))
        hook_handles.append(hook)


# %%

# 迭代DataLoader并打印每个批次
if num_batches_to_iterate ==0:
    num_batches_to_iterate = len(dataloader)

for batch_num, batch in enumerate(dataloader):
    Batch_num+=1
    print(f"Batch number: {batch_num}")
    if batch_num >= num_batches_to_iterate:
        break
    batch_input_ids, batch_attention_mask = batch
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_mask = batch_attention_mask.to(device)
    with torch.no_grad():
        output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    #print(output)
    torch.cuda.empty_cache()
    print("/n/n/n")

for hook in hook_handles:
    hook.remove()

# %%

# %%
if mode:
    folder_path = os.path.join(folder_path,"orgin")
    if item=="both":
        np.savez(os.path.join(folder_path,'orgin_input.npz'), **orgin_input)
        np.savez(os.path.join(folder_path,'orgin_weight.npz'), **orgin_weight)
        np.savez(os.path.join(folder_path,'orgin_output.npz'), **orgin_output)
    elif item=="input":
        np.savez(os.path.join(folder_path,'orgin_input.npz'), **orgin_input)
    elif item=="weight":
        np.savez(os.path.join(folder_path,'orgin_weight.npz'), **orgin_weight)
    elif item=="output":
        np.savez(os.path.join(folder_path,'orgin_output.npz'), **orgin_output)
else:
    if item=="both":
        np.savez(os.path.join(folder_path,'binary_input_16bit.npz'), **binary_input)
        np.savez(os.path.join(folder_path,'binary_weight_8bit.npz'), **binary_weight)
        np.savez(os.path.join(folder_path,'binary_output_16bit.npz'), **binary_output)
    elif item=="input":
        np.savez(os.path.join(folder_path,'binary_input_16bit.npz'), **binary_input)
    elif item=="weight":
        np.savez(os.path.join(folder_path,'binary_weight_8bit.npz'), **binary_weight)
    elif item=="output":
        np.savez(os.path.join(folder_path,'binary_output_16bit.npz'), **binary_output)


