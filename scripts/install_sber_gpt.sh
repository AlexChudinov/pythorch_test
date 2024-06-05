apt install clang llvm llvm-dev
apt install python3-packaging
git clone https://github.com/qywu/apex
pip3 install --upgrade pip
pip3 install packaging
pip3 install -v --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
