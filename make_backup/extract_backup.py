import os
import tarfile
import argparse
import logging

def setup_logging():
    # Define o formato das mensagens de log
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configura o logging para registrar mensagens no arquivo 'extraction.log'
    logging.basicConfig(filename='extraction.log', level=logging.INFO, format=log_format)
    
    # Cria um manipulador de log para a saída do console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Adiciona o manipulador ao logger principal
    logging.getLogger().addHandler(console_handler)

def extract_tar_gz_files(directory=".", max_files=None):
    """
    Extract up to max_files .tar.gz files in the specified directory.
    """
    extracted_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tar.gz"):
                filepath = os.path.join(root, file)
                
                logging.info(f"Iniciando a extração do arquivo: {filepath}")
                
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=root)
                
                extracted_count += 1
                
                logging.info(f"Extração do arquivo {file} concluída ({extracted_count} de {max_files if max_files else 'todos'}).")
                
                if max_files and extracted_count >= max_files:
                    logging.info("Limite máximo de arquivos extraídos alcançado.")
                    return

    logging.info("Todas as extrações foram concluídas.")

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Extract .tar.gz files.")
    parser.add_argument("--max", type=int, help="Maximum number of files to extract.")
    args = parser.parse_args()
    
    extract_tar_gz_files(max_files=args.max)