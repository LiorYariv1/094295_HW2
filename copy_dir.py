import os
import shutil

# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")




if __name__ == "__main__":
    print("start")
    # src = "data/train"
    # dst = "data/train(oldcopy)"
    # # shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False,
    # #                 dirs_exist_ok=False)
    # shutil.copytree(src, dst)

    src = "data/trial/train_val_loss_plot1.png"
    dst = "data/trial2"
    # shutil.copy(src, dst)
    shutil.move(src, dst)
    print("done")


