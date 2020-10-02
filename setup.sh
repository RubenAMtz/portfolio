sudo pip uninstall jupyter-client==6.1.7 
sudo pip uninstall jupyter-core==4.6.3 
sudo pip uninstall jupyterlab-pygments==0.1.2
sudo pip uninstall markdown==3.2.2
sudo pip uninstall ipykernel==5.3.4
sudo pip uninstall tensorboard==2.3.0
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml