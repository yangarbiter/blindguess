
from package.variables import auto_var
from utils import setup_experiments


def run_exp_name(auto_var):
    pass


def main():
    setup_experiments(auto_var)
    auto_var.parse_argparse()

if __name__ == '__main__':
    main()
