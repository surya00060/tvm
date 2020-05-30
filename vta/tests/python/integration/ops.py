import re


def find_substring(string,start_char,end_char):
    substrings = []

    sub = ""
    for i in range(len(string)):
        
        if(string[i] == start_char):
            while(string[i] != end_char and i < len(string)):
                sub += string[i]
                i = i+1
            if(string[i] == end_char):
                substrings.append(sub)
                sub = ""
    return substrings




class RelayNode:
    def __init__(self,name,size,last_used,op = None,dram_addr = 0):
        self.name = name
        self.size = size
        self.dram_addr = dram_addr
        self.last_used = last_used
    


class conv2d:
    def __init__(self,line):
        self.parse(line)

    def parse(self,line):
        self.output = find_substring(line,'%','=')[0]
        line = line.replace(self.output,"")
        self.input = find_substring(line,'%',',')[0]
        self.weight = find_substring(line,'%',',')[1]
        
        return 0,0,0,0


line =  " %0 = nn.conv2d(%input0, %weight.1, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 64, 64), float32] */;"

c = conv2d(line)
