from asr.collect import chdir


def test_workflow():
    with chdir('Si'):
        from asr.workflow import main
        main()


if __name__ == '__main__':
    test_workflow()
