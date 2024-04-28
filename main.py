import utils.fflow as flw
import torch

def main():
    option = flw.read_option()
    flw.setup_seed(option['seed'])
    server = flw.initialize(option)
    try:
        server.run()
    except:
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()