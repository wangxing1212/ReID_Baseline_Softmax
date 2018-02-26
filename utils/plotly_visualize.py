import plotly
plotly.offline.init_notebook_mode(connected=True)
from plotly.offline import iplot
from plotly.graph_objs import Scatter, Layout
from IPython.display import clear_output
from config import opt

class Plotly_with_Update(object):
    def __init__(self,y):
        self.y = dict.fromkeys(y.keys(),[])
        for k,v in y.items():
            if isinstance(v, list):
                self.y[k] = v
            else:
                self.y[k] = [v]
        
    def plot(self):
        trace = []
        for k,v in self.y.items():
            x = list(range(0,len(self.y[k])))
            trace.append(Scatter(x = x,y=self.y[k],mode = 'lines+markers',name = k))

        fig = {'data' : trace,'layout' : Layout(title="hello world")}
        iplot(fig)
    
    def update(self,new_value,**kwargs):
        for k,v in new_value.items():
            if isinstance(v, list):
                self.y[k] = self.y[k]+v
            else:
                self.y[k] = self.y[k]+[v]
        
        clear_output()
        opt.parse(kwargs,show_config=True)
        