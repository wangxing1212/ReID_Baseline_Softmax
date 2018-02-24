from config import opt
from main import train
from main import test
import ipywidgets as widgets
from IPython.display import display

def train_test_demo():
    dataset_name = widgets.Dropdown(
                        options=['Market1501', 'DukeMTMC'],
                        value='Market1501',
                        description='Dataset:',
                        disabled=False,
                    )
    model = widgets.Dropdown(
                        options=['ResNet50', 'DenseNet121'],
                        value='ResNet50',
                        description='Model:',
                        disabled=False,
                    )

    re_ranking = widgets.Checkbox(
                    value=True,
                    description='Re_Ranking:',
                    disabled=False
                )


    train_button = widgets.Button(description="Train")
    test_button = widgets.Button(description='Test')
    items = [train_button,test_button]
    button = widgets.Box(items)

    opt.dataset_name = dataset_name.value
    opt.model = model.value
    opt.re_ranking = re_ranking.value
    display(dataset_name,model,re_ranking,button)
    def click_train(b):
        train()
    def click_test(b):
        test()
    train_button.on_click(click_train)
    test_button.on_click(click_test)