# Usa uma imagem base do Miniconda
FROM continuumio/miniconda3

# Copia o arquivo environment.yml para o contêiner
COPY environment.yml /tmp/environment.yml

# Cria o ambiente Conda
RUN conda env create -f /tmp/environment.yml

# Ativa o ambiente Conda automaticamente
RUN echo "source activate py37" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Instala o IPyKernel e cria um kernel específico para o ambiente
RUN /opt/conda/envs/myenv/bin/python -m ipykernel install --user --name py37 --display-name "Python (py37)"

# Define o diretório de trabalho
WORKDIR /workspace

# Mantém o contêiner em execução
CMD ["tail", "-f", "/dev/null"]