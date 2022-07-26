# torch-compression

## Project Installation
    `sh install.sh`

## How to run?
### Checklist before running
* Modify the checkpoint path in `example/util/log_manage.py`.
* Make sure `[model].py` can find the dataset correctly.

### Training
* To train the example coder, run the command below:
  ```
  $ python3 [model].py train
  ```

* To change the target bit-per-pixel, please specify the lambda at beginning:
  ```
  $ python3 [model].py train --lambda=<specified lambda value>