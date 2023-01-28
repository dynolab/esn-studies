import os
import time
from fabric.api import * # lcd, local, run, env, put, get

def hello(): #работает!
    """
    print Hello!
    """
    print("Hello!")


def cd_to_bru():
    """
    move to Brusselator directory or create one through ssh 
    """
    list_ls_bru = run ("ls")
    home_path = run('pwd')
    path_dir = f'{home_path}/Brusselator'
    #change this path to yours!
    path_to_needed = 'C:/Users/njuro/Documents/restools/Researches/2022-07-26-predicting-brusselator-via-esn'
    need_in_bru = [f'1-Dataset_from_Calum_1D_Brusselator']
    if 'Brusselator' not in list_ls_bru: 
        print('We need Brusselator dir ') 
        run(f"mkdir {path_dir}")
        with cd(f'{path_dir}'):
            list_ls = run ("ls")
        for item in need_in_bru:
            if item not in list_ls:
                put(f'{path_to_needed}/{item}' , f'{path_dir}')
        run(f"mkdir {path_dir}/Programs")
        run(f"mkdir {path_dir}/Figures")
  #  run('ls')
  #  with cd(f'{path_dir}'):
  #      run('ls')
    #находиться надо в папке Brusselator
    return path_dir

   
def run_test(prog_changed=False):
    """
    temporary for tests
    """
    env.user = 'amamelkina' #<- this is what you need for test, then copy:
    list_ls_bru = run ("ls")
    home_path = run('pwd')

    path_dir = cd_to_bru()
    path_to_figure = 'C:/Users/njuro/Documents/Диплом Магистратура/Figures/1D_bru_images_hyperparameters'
    get(remote_path=f'{path_dir}/Figures/hyperparameter_search.png', local_path=f'{path_to_figure}/hyperparameter_search.png')
    print(f"\nResult is in {path_to_figure}")
   # put('C:/Users/njuro/Documents/esn-studies/requirements.txt' , f'{home_path}')
    print("Tada")



env.hosts = ['hpc.rk6.bmstu.ru']
#env.user = 'amamelkina' # Set the username

def run_hyper_search(prog_changed):
    """
    run Hyperparameter search through ssh

    run fabfile with parameters: fab hello:param="Hello world!" (if there is more than one parameter, put them with ',' and no ' ')

    prog_changed -- if you changed program (bru_2_hyp_search_for_ssh.py), you need to put changes on server.
        Set prog_changed = True and program will be updated on the server

    
    """
    
    #ввод хоста и пользователя, если это не я, через prompt (если я, то просто Enter)
    user = prompt("Enter username", default='amamelkina')
    env.user = user # Set the username        
  #  print(env.user, env.hosts)

    #перейти в нужную директорию, если она есть, перекинуть, если ее нет
    path_dir = cd_to_bru()
  #  print('path_dir ', path_dir)
    #перемещаем измененные программы, если надо
    if (prog_changed == "True") :
        print("Updating program... \n")
        put(f'{os.getcwd()}/bru_2_hyp_search_for_ssh.py' , f'{path_dir}/Programs/bru_2_hyp_search_for_ssh.py')

    with cd(f'{path_dir}/Programs'):
  #      run('ls')
        #запускаем программу
        run('python3 bru_2_hyp_search_for_ssh.py')
       
        print(f"TADAAaa")
  #      run('ls')
        
    #копируем результат себе
    #измени этот путь на тот, где хочешь сохранить результат!
    path_to_figure = 'C:/Users/njuro/Documents/Диплом Магистратура/Figures/1D_bru_images_hyperparameters'
    get(remote_path=f'{path_dir}/Figures/hyperparameter_search.png', local_path=f'{path_to_figure}/hyperparameter_search.png')
    print(f"\nResult is in {path_to_figure}")



#ssh - 
#my  - 
