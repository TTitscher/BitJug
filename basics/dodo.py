
def task_2014_Liu():
    return {
        "file_dep": ["2014_Liu.py"],
        "actions": ["urup 2014_Liu.py -o 2014_Liu.md", "pandoc --css /home/schmaus/dotfiles/github.css --toc --standalone --mathjax 2014_Liu.md -o 2014_Liu.html"]
    }
    
