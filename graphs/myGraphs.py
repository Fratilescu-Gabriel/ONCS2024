import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
DEFAULT_COLORS = ['b', 'g', 'r', 'y', 'c', 'm', 'k']

class graph():
    def __init__(self, axis_number, *args, **kwargs) -> None:
        self.axis_number = axis_number
        self.axis_labels = []
        self.functions = []
        self.accessories = []
        self.color = DEFAULT_COLORS[0]
        self.grid = None
        self.legend = None
        
        self.update_parameters(kwargs=kwargs)
        
                
    def add_function(self, function, *args, **kwargs):
        for axis in range(0, self.axis_number):
            self.functions.append(function[axis])
        
        item_accessory = {}
        
        for key, value in kwargs.items():
            item_accessory[key] = value 
        
        self.accessories.append(item_accessory)
        
    
    def update_parameters(self, *args, **kwargs):
        for key, value in kwargs['kwargs'].items():
            if key == 'axis_labels':
                self.axis_labels = value
            elif key == 'grid':
                self.grid = value
            elif key == 'legend':
                self.legend = value
    
    def plot_figure(self, delete_functions = False, *args, **kwargs):
        plt.figure()
        
        for index in range(0,len(self.functions), self.axis_number):
            self.plot_graph(index)
        
        if self.grid:
            plt.grid()
        
        if self.legend:
            plt.legend()
        
        plt.xlabel = self.axis_labels[0]
        
        plt.ylabel = self.axis_labels[1]
        
        plt.show()
        
        if delete_functions:
            self.functions = []   
            self.accessories = []     
    
    
    
    def plot_graph(self, index, *args, **kwargs):
        color_change = False
        color = self.color
        label = None
        star = ''
        print(index, self.axis_number)
        for key, value in self.accessories[int(index/self.axis_number)].items():
            if key == 'color':
                color = value
                color_change = True
            elif key == 'label':
                label = value
            elif key == 'star':
                star = value
        
        if self.axis_number == 2:
            plt.plot(self.functions[index],self.functions[index+1], color = color, label = label)
        
        if not color_change:
            if DEFAULT_COLORS.index(color) + 1 <= len(DEFAULT_COLORS) - 1:
                self.color = DEFAULT_COLORS[DEFAULT_COLORS.index(color)+1]
            else:
                self.color = DEFAULT_COLORS[0]
            
        
        
        
        
        
        
    
            
x = np.linspace(0,20,30)
y = np.linspace(10,40,30)
z = np.linspace(20,50,40)
x1 = np.linspace(0,50,40)
y1 = np.linspace(10,70,40)
z1 = np.linspace(40,50,40)
        
gr = graph(2, axis_labels = ['fuck up', 'fuck down'], grid = True, legend = True)
gr.add_function([x,y], color = 'r', label = 'plot1')
gr.add_function([x1,y1], label = 'plot2')
gr.add_function([z,z1], 'g*', label = 'test')
a = gr.functions
print(a)
gr.plot_figure(delete_functions= True)


    
