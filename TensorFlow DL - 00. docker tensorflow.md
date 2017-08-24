<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Docker TensorFlow 설치

도커 터미널 오픈  

```
docker pull tensorflow/tensorflow
```

설치 완료 후 다음 명령 실행
```
sudo docker run -it -p 8888:8888 -p 6006:6006 -v /c/Users/Administrator/Docker/work:/home/testu/work --name tftest tensorflow/tensorflow
```
8888 포트는 Jupyter 용  
6006 포트는 TensorBoard 용


실행후 나오는 메시지
```
Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=6053d1ff7f865ada23adc7938c28fc7733a25f2f06b1dbc6
```

브라우저로 192.168.99.100:8888 접속  
토큰값 "6053d1ff7f865ada23adc7938c28fc7733a25f2f06b1dbc6" 으로 로그인
/notebooks/ 폴더가 기본 폴더로 나옴


참조 링크  
윈도우 도커 TF설치 (http://solarisailab.com/archives/384)  
도커 TF설치 (http://bryan7.tistory.com/763)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### sudo 설치

열린 도커 터미널에서 tftest 컨테이너 중지
```
ctrl+c
```

tftest 재실행 및 우분투 버전 확인

```
docker start tftest
docker exec -it tftest bash

cat /etc/issue
cat /etc/*release
```

sudo 설치
```
apt-get update
apt-get install sudo
sudo
cat /etc/sudoers
```

참조 링크  
리눅스 sudo 사용자 추가 (http://webdir.tistory.com/255)  
우분투 계정에 sudo 권한 부여 (http://sarghis.com/blog/856/)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### vim 설치


```
apt-get install vim
```

설치 완료 후 환경 설정
```
vim ~/.vimrc
```

i (현재 커서 위치에 글자 입력) 누르고 다음 코드 입력
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
esc 누르고
:wq (수정 적용하고 나옴) 쓰고 나옴


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Jupyter 패스워드 로그인

토큰 정보 대신 패스워드로 로그인하기


docker tftest 실행

```
docker start tftest
docker exec -it tftest bash

jupyter notebook --help
```


root/.jupyter 로 가서 ipython 실행하기
```
cd /root/.jupyter/
ls -als

ipython
```

다음 코드 입력
```
In [1]: from IPython.lib import passwd
In [2]: passwd()
Enter password: a123
Verify password: a123
Out[2]: 'sha1:fff18f9598e2:b4b9d8a4b7c957ef57ae5a3562144722484ec5d6'
In [3]: ctrl+d
Do you really want to exit ([y]/n)? y
```

위에서 나온 키값을 jupyter 설정 파일에 복사  
(/root/.jupyter/jupyter_notebook_config.py)
```
vim jupyter_notebook_config.py
```
vim 으로 열면 다음 코드 부분있어
```
...
c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False
c.MultiKernelManager.default_kernel_name = 'python2'
...
```
위 코드 밑에 아래 라인 추가 (키값은 ipython 에서 받은 키 값 적용)
```
c.NotebookApp.password = 'sha1:fff18f9598e2:b4b9d8a4b7c957ef57ae5a3562144722484ec5d6'
```

tftest 재실행
```
exit
docker stop tftest
docker start tftest
docker exec -it tftest bash
```

브라우저에서 jupyter 열면 패스워드 입력으로 바뀜  
http://192.168.99.100:8888/login

<br />
참조 링크  
패스워드로 로그인 (https://financedata.github.io/posts/jupyter-notebook-authentication.html#4.-jupyter-notebook-%EC%8B%A4%ED%96%89)


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Virtualenv 설치 (Optional)

https://beomi.github.io/2016/12/28/HowToSetup-Virtualenv-VirtualenvWrapper/


터미널
```
docker start tftest bash
docker exec -it tftest bash

cd /
which python
python -V

pip로 깔린 모든 패키지의 버전 리스트확인
pip freeze

tensorflow==1.2.1
```

Ubuntu의 경우에는 14버전 기준으로 Python2와 Python3이 기본적으로 설치되어있습니다.  
하지만 pip/pip3이 설치되어있지 않을 수 있기 때문에 python-pip나 python3-pip를 설치해야 합니다.

```
# APT를 업데이트
sudo apt-get update && apt-get upgrade -y
# Python2를 이용할 경우
sudo apt-get install python-pip python-dev
# Python3을 이용할 경우
sudo apt-get install python3-pip python3-dev

pip 설치 확인
# Python2 pip의 경우
pip -V
pip install --upgrade pip

# Python3 pip의 경우
pip3 -V
pip3 install --upgrade pip
```
이제 pip설치가 완료되었으므로 Virtualenv와 VirtualenvWrapper를 설치해보겠습니다.
```
# Python2의 경우
pip install virtualenv virtualenvwrapper
# Python3의 경우
pip3 install virtualenv virtualenvwrapper
```
 

#### Virtualenv

$ virtualenv --python=파이썬버전 가상환경이름
```
cd /home/testu
virtualenv --python=python3.5 py3_env
virtualenv --python=python2.7 test_env2
```
현재 폴더내에(/home/testu)
 py3_env, test_env2 폴더 생성됨

Python3이 설치된 가상환경 py3_env로 진입한 경우
```
source py3_env/bin/activate
python -V
deactivate
```

지우기
```
rm -R py3_env
rm -R test_env2
```

#### VirtualenvWrapper 설정
VirtualEnv를 사용하기 위해서는 source를 이용해 가상환경에 진입합니다.
그러나, 이 진입 방법은 가상환경이 설치된 위치로 이동해야되는 것 뿐 아니라 
가상환경이 어느 폴더에 있는지 일일이 사용자가 기억해야 하는 단점이 있습니다.
이를 보완하기 위해 VirtualenvWrapper를 사용합니다.
또한, VirtualenvWrapper를 사용할 경우 터미널이 현재 위치한 경로와 관계없이 
가상환경을 활성화할 수 있다는 장점이 있습니다.

VirtualenvWrapper는 .bashrc나 .zshrc에 약간의 설정과정을 거쳐야 합니다.

우선 홈 디렉토리로 이동
가상환경이 들어갈 폴더 .virtualenvs를 만들어주세요.
```
cd ~
mkdir ~/.virtualenvs

vim ~/.bashrc
```
파일 제일 마지막에 아래 코드를 복사해 붙여넣어줍시다. (파일이 없다면 만들어 사용하시면 됩니다.)
```
# python virtualenv settings
export WORKON_HOME=~/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON="$(command \which python3)" # Usage of python3
source /usr/local/bin/virtualenvwrapper.sh
```
파일 경로 다른경우
find /usr -name virtualenvwrapper.sh 로 찾아서 수정

```
source ~/.bashrc
```

#### VirtualenvWrapper 명령어들

가상환경 만들기  
$ mkvirtualenv 가상환경이름
```
mkvirtualenv test_env3
mkvirtualenv --python=`which python3` env3

virtualenv --python=python2.7 test_env2
```
/root/.virtualenvs 안에 test_env3, env3 폴더 생성됨


가상환경 진입하기 + 가상환경 목록 보기
$ workon 가상환경이름
가상환경으로 진입시 앞에 (가상환경이름)이 붙습니다.
(가상환경이름) $

```
workon env3
python
import tensorflow as tf
```

가상환경 빠져나오기
```
deactivate
```

이거 보고 설치해보자  
https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/get_started/os_setup.html


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### Git 설치

```
sudo add-apt-repository ppa:git-core/ppa
sudo apt-get update && sudo apt-get dist-upgrade
sudo apt-get install git-core
 
git version

cd /home/testu
```
참조 링크  
http://thisblogbusy.tistory.com/entry/%EC%9A%B0%EB%B6%84%ED%88%AC-1604%EC%97%90%EC%84%9C-GIT-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0


<br /><br />
<!--------------------------------------------------------------->
<!--------------------------------------------------------------->

### 컨테이너 백업


tftest 컨테이너가 존재하는지 확인
```
sudo docker ps -a | grep tftest
```

tftest 컨테이너를  'tf_test' 라는 이미지로 저장
```
sudo docker commit tftest tf_test
```

tftest_backup.tar 라는 이름으로 백업
```
sudo docker save tftest_backup > /c/Users/Administrator/Docker/tf_test.tar
```

복원시
```
docker load < /c/Users/Administrator/Docker/tf_test.tar

sudo docker run -it -p 8888:8888 -p 6006:6006 -v /c/Users/Administrator/Docker/work:/home/testu/work --name tftest tf_test

docker start tftest
docker exec -it tftest bash
```

참조 링크  
http://digndig.kr/docker/709/

