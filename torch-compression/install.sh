echo "$(pwd)" >> ./torch_compression.pth
SITE=$(python -s -c 'import site; print(site.USER_SITE)')
#SITE="/home/tl32rodan/.conda/envs/canfvc/lib/python3.6/site-packages"
mv ./torch_compression.pth "$SITE/torch_compression.pth"

python -m pip install -r requirements.txt
cd torch_compression/torchac/
python setup.py install --user
rm -rf build dist torchac_backend_cpu.egg-info
cd ../../
