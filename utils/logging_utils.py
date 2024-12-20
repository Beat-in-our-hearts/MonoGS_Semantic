import rich
from utils.semantic_setting import Semantic_Config

_log_styles = {
    "MonoGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="MonoGS"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)

def info(text):
    rich.print(f"[Info]{text}")
    with open(Semantic_Config.log_file, 'a+') as f:
        f.write(f"{text}\n")
    
def debug(text):
    if Semantic_Config.Debug:
        print(text)
    with open(Semantic_Config.log_file, 'a+') as f:
        f.write(f"{text}\n")