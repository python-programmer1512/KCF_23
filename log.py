import logging


class log():
    def __init__(self,path):
        self.path=path
        self.f=open(path,'w')
        self.f.close()

    def open(self):
        self.f=open(self.path,'w')
    def close(self):
        self.f.close()

    def print_log(self,input):
        self.f.write(input+"\n")
        print(input)
