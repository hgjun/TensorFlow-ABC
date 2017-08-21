<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Docker TensorFlow ��ġ

��Ŀ �͹̳� ����  

```
docker pull tensorflow/tensorflow
```

��ġ �Ϸ� �� ���� ��� ����
```
sudo docker run -it -p 8888:8888 -p 6006:6006 -v /c/Users/Administrator/Docker/work:/home/testu/work --name tftest tensorflow/tensorflow
```
8888 ��Ʈ�� jupyter ��  
6006 ��Ʈ�� ���߿� TensorBoard ���� ���


������ ������ �޽���
```
Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=6053d1ff7f865ada23adc7938c28fc7733a25f2f06b1dbc6
```

�������� 192.168.99.100:8888 ����  
��ū�� "6053d1ff7f865ada23adc7938c28fc7733a25f2f06b1dbc6" ���� �α���
/notebooks/ ������ �⺻ ������ ����


���� ��ũ  
������ ��Ŀ TF��ġ (http://solarisailab.com/archives/384)  
��Ŀ TF��ġ (http://bryan7.tistory.com/763)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### sudo ��ġ

���� ��Ŀ �͹̳ο��� tftest �����̳� ����
```
ctrl+c
```

tftest ����� �� ����� ���� Ȯ��

```
docker start tftest
docker exec -it tftest bash

cat /etc/issue
cat /etc/*release
```

sudo ��ġ
```
apt-get update
apt-get install sudo
sudo
cat /etc/sudoers
```

���� ��ũ  
������ sudo ����� �߰� (http://webdir.tistory.com/255)  
����� ������ sudo ���� �ο� (http://sarghis.com/blog/856/)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### vim ��ġ


```
apt-get install vim
```

��ġ �Ϸ� �� ȯ�� ����
```
vim ~/.vimrc
```

i (���� Ŀ�� ��ġ�� ���� �Է�) ������ ���� �ڵ� �Է�
```
set number
set ai
set si
set cindent
set shiftwidth=4
set tabstop=4
set ignorecase
set hlsearch
set expandtab
set background=dark
set nocompatible
set fileencodings=utf-8,euc-kr
set bs=indent,eol,start
set history=1000
set ruler
set nobackup
set title
set showmatch
set nowrap
set wmnu
syntax on
```
esc ������
:wq (���� �����ϰ� ����) ���� ����


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Jupyter �н����� �α���

��ū ���� ��� �н������ �α����ϱ�


docker tftest ����

```
docker start tftest
docker exec -it tftest bash

jupyter notebook --help
```


root/.jupyter �� ���� ipython �����ϱ�
```
cd /root/.jupyter/
ls -als

ipython
```

���� �ڵ� �Է�
```
In [1]: from IPython.lib import passwd
In [2]: passwd()
Enter password: a123
Verify password: a123
Out[2]: 'sha1:fff18f9598e2:b4b9d8a4b7c957ef57ae5a3562144722484ec5d6'
In [3]: ctrl+d
Do you really want to exit ([y]/n)? y
```

������ ���� Ű���� jupyter ���� ���Ͽ� ����  
(/root/.jupyter/jupyter_notebook_config.py)
```
vim jupyter_notebook_config.py
```
vim ���� ���� ���� �ڵ� �κ��־�
```
...
c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False
c.MultiKernelManager.default_kernel_name = 'python2'
...
```
�� �ڵ� �ؿ� �Ʒ� ���� �߰� (Ű���� ipython ���� ���� Ű �� ����)
```
c.NotebookApp.password = 'sha1:fff18f9598e2:b4b9d8a4b7c957ef57ae5a3562144722484ec5d6'
```

tftest �����
```
exit
docker stop tftest
docker start tftest
docker exec -it tftest bash
```

���������� jupyter ���� �н����� �Է����� �ٲ�  
http://192.168.99.100:8888/login

<br />
���� ��ũ  
�н������ �α��� (https://financedata.github.io/posts/jupyter-notebook-authentication.html#4.-jupyter-notebook-%EC%8B%A4%ED%96%89)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Virtualenv ��ġ (Optional)

https://beomi.github.io/2016/12/28/HowToSetup-Virtualenv-VirtualenvWrapper/


�͹̳�
```
docker start tftest bash
docker exec -it tftest bash

cd /
which python
python -V

pip�� �� ��� ��Ű���� ���� ����ƮȮ��
pip freeze

tensorflow==1.2.1
```

Ubuntu�� ��쿡�� 14���� �������� Python2�� Python3�� �⺻������ ��ġ�Ǿ��ֽ��ϴ�.  
������ pip/pip3�� ��ġ�Ǿ����� ���� �� �ֱ� ������ python-pip�� python3-pip�� ��ġ�ؾ� �մϴ�.

```
# APT�� ������Ʈ
sudo apt-get update && apt-get upgrade -y
# Python2�� �̿��� ���
sudo apt-get install python-pip python-dev
# Python3�� �̿��� ���
sudo apt-get install python3-pip python3-dev

pip ��ġ Ȯ��
# Python2 pip�� ���
pip -V
pip install --upgrade pip

# Python3 pip�� ���
pip3 -V
pip3 install --upgrade pip
```
���� pip��ġ�� �Ϸ�Ǿ����Ƿ� Virtualenv�� VirtualenvWrapper�� ��ġ�غ��ڽ��ϴ�.
```
# Python2�� ���
pip install virtualenv virtualenvwrapper
# Python3�� ���
pip3 install virtualenv virtualenvwrapper
```
 

#### Virtualenv

$ virtualenv --python=���̽���� ����ȯ���̸�
```
cd /home/testu
virtualenv --python=python3.5 py3_env
virtualenv --python=python2.7 test_env2
```
���� ��������(/home/testu)
 py3_env, test_env2 ���� ������

Python3�� ��ġ�� ����ȯ�� py3_env�� ������ ���
```
source py3_env/bin/activate
python -V
deactivate
```

�����
```
rm -R py3_env
rm -R test_env2
```

#### VirtualenvWrapper ����
VirtualEnv�� ����ϱ� ���ؼ��� source�� �̿��� ����ȯ�濡 �����մϴ�.
�׷���, �� ���� ����� ����ȯ���� ��ġ�� ��ġ�� �̵��ؾߵǴ� �� �� �ƴ϶� 
����ȯ���� ��� ������ �ִ��� ������ ����ڰ� ����ؾ� �ϴ� ������ �ֽ��ϴ�.
�̸� �����ϱ� ���� VirtualenvWrapper�� ����մϴ�.
����, VirtualenvWrapper�� ����� ��� �͹̳��� ���� ��ġ�� ��ο� ������� 
����ȯ���� Ȱ��ȭ�� �� �ִٴ� ������ �ֽ��ϴ�.

VirtualenvWrapper�� .bashrc�� .zshrc�� �ణ�� ���������� ���ľ� �մϴ�.

�켱 Ȩ ���丮�� �̵�
����ȯ���� �� ���� .virtualenvs�� ������ּ���.
```
cd ~
mkdir ~/.virtualenvs

vim ~/.bashrc
```
���� ���� �������� �Ʒ� �ڵ带 ������ �ٿ��־��ݽô�. (������ ���ٸ� ����� ����Ͻø� �˴ϴ�.)
```
# python virtualenv settings
export WORKON_HOME=~/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON="$(command \which python3)" # Usage of python3
source /usr/local/bin/virtualenvwrapper.sh
```
���� ��� �ٸ����
find /usr -name virtualenvwrapper.sh �� ã�Ƽ� ����

```
source ~/.bashrc
```

#### VirtualenvWrapper ��ɾ��

����ȯ�� �����  
$ mkvirtualenv ����ȯ���̸�
```
mkvirtualenv test_env3
mkvirtualenv --python=`which python3` env3

virtualenv --python=python2.7 test_env2
```
/root/.virtualenvs �ȿ� test_env3, env3 ���� ������


����ȯ�� �����ϱ� + ����ȯ�� ��� ����
$ workon ����ȯ���̸�
����ȯ������ ���Խ� �տ� (����ȯ���̸�)�� �ٽ��ϴ�.
(����ȯ���̸�) $

```
workon env3
python
import tensorflow as tf
```

����ȯ�� ����������
```
deactivate
```

�̰� ���� ��ġ�غ���  
https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/get_started/os_setup.html


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Git ��ġ

```
sudo add-apt-repository ppa:git-core/ppa
sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install git-core
 
git version

cd /home/testu
```
���� ��ũ  
http://thisblogbusy.tistory.com/entry/%EC%9A%B0%EB%B6%84%ED%88%AC-1604%EC%97%90%EC%84%9C-GIT-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### �����̳� ���


tftest �����̳ʰ� �����ϴ��� Ȯ��
```
sudo docker ps -a | grep tftest
```

tftest �����̳ʸ�  'tf_test' ��� �̹����� ����
```
sudo docker commit tftest tf_test
```

tftest_backup.tar ��� �̸����� ���
```
sudo docker save tftest_backup > /c/Users/Administrator/Docker/tf_test.tar
```

������
```
docker load < /c/Users/Administrator/Docker/tf_test.tar

sudo docker run -it -p 8888:8888 -p 6006:6006 -v /c/Users/Administrator/Docker/work:/home/testu/work --name tftest tf_test

docker start tftest
docker exec -it tftest bash
```

���� ��ũ  
http://digndig.kr/docker/709/

