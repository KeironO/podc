# Interpreting the real-time dynamic 'sliding sign' and predicting pouch of Douglas obliteration using deep learning

## Introduction

## Getting Set up

For ease of management, I have used ```pipenv``` to setup the project. I hope that this works across all platforms, but I am unable to test against Apple macOS or Microsoft Windows. If the project fails to build correctly, feel free to submit an Issue and I will try and help you - time willing.

If you've already set up ```pipenv``` before,  ignore the next step. If you haven't then type in the following command into your terminal emulator:

```bash
pip3 install pipenv --user
```

This should set up ```pipenv``` in userspace.

Now, clone the GitHub repository:

```bash
git clone https://www.github.com/KeironO/podc
```

Change your present working directory to the cloned repository, and type:

```bash
pipenv sync
```

This should now get all of the projects dependencies. Once finished type:

```bash
pipenv shell
```

And you should be able to run all of the files in the ```podc``` directory.

## License

Code released under the GNU General Public License v2 and later.