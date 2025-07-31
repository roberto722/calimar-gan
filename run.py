if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    config_file = sys.argv[2] if len(sys.argv) > 2 else "configs/train_config.json"
    from main import main
    main(mode, config_file)
