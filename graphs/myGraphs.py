import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DEFAULT_COLORS = ['b-', 'g-', 'r-', 'y-', 'c-', 'm-', 'k-']

class graph():
    def __init__(self, axis_number, *args, **kwargs) -> None:
        self.axis_number = axis_number
        self.axis_labels = [None]*self.axis_number
        self.functions = None
        self.color = DEFAULT_COLORS[0]
        self.label = None
        self.star = None
        self.grid = None
        self.legend = None
        
        for key, value in kwargs.items():
            if(key == 'color'):
                self.color = value
            if(key == 'label'):
                self.label = value
            if(key == 'star'):
                self.star = value
            if(key == 'grid'):
                self.grid = value
            if(key == 'legend'):
                self.legend = value
                
    def add_function(self, function):
        if  isinstance(self.functions, list):
            self.functions.append(function)
        else:
            self.functions = [[[None]*np.size(function, axis=1)]*self.axis_number]
            self.functions[0] = function
            
x = np.linspace(0,50,2)
y = np.linspace(10,70,2)
z = np.linspace(40,50,2)
x1 = np.linspace(0,50,3)
y1 = np.linspace(10,70,3)
z1 = np.linspace(40,50,3)
        
gr = graph(3)
gr.add_function(np.array([x,y,z]))
gr.add_function(np.array([x1,y1,z1]))
a = gr.functions
print(a)
    
