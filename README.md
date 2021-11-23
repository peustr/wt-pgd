# wt-pgd

## Installation

```bash
conda create -n wtpgd python=3.7
conda activate wtpgd
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
pip install -e .
```

## Example usage

Getting the loss landscape of an adversarial defence.
```python
# assuming the model and data has been loaded
x, y, z, g1, g2 = get_loss_landscape(model, img, target)
fig, ax = create_figure(x, y, z, savefig=True)
plt.show()
```

Getting the loss landscape of an adversarial defence under attack by WTPGD.
```python
# assuming the model and data has been loaded
x, y, z, g1, g2 = get_loss_landscape(model, img, target, use_wtpgd=True, wtpgd_args={})
fig, ax = create_figure(x, y, z, savefig=True)
plt.show()
```
