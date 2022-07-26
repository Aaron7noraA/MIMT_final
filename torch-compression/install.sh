echo "$(pwd)" >> ./torch_compression.pth
SITE=$(python3 -c 'import site; print(site.USER_SITE)')
mv ./torch_compression.pth "$SITE/torch_compression.pth"

pip3 install -r requirements.txt
cd torch_compression/torchac/
python3 setup.py install --user
rm -rf build dist torchac_backend_cpu.egg-info
cd ../../
