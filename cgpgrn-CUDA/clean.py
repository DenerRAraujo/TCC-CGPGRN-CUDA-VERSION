import os
import shutil
import glob

def clean_directory():
    # 1. Pastas a serem removidas inteiramente
    folders_to_remove = [
        "executions_parallel",
        "kmeans",
        "time_counting"
    ]

    # 2. Padrões de arquivos a serem removidos
    file_patterns = [
        "BKM_*",                # Arquivos BKM (BKM_GSD_full_pt0, etc)
        "CGPGRN_parameters*",   # Arquivos de parâmetros
        "CGPGRN_times*",        # Arquivos de tempos
        "LogFile_*",            # Arquivos de log
        "splineData_*",         # Arquivos de spline
        "*.sh",                 # Todos os scripts shell
        "*_exe_*",              # Scripts de execução paralela (ex: GSD_exe_1_Cluster0_pt0)
        "*_full"                # Arquivos de execução full (ex: GSD_full)
    ]

    print("🧹 Iniciando limpeza do diretório...\n")

    # Remover pastas
    for folder in folders_to_remove:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"  [PASTA REMOVIDA]  {folder}")
            except Exception as e:
                print(f"  [ERRO PASTA]     Não foi possível remover {folder}: {e}")

    # Remover arquivos baseados em padrões (wildcards)
    for pattern in file_patterns:
        files = glob.glob(pattern)
        for file in files:
            # Preserva os scripts Python principais e arquivos do sistema
            if os.path.isfile(file):
                try:
                    os.remove(file)
                    print(f"  [ARQUIVO REMOVIDO] {file}")
                except Exception as e:
                    print(f"  [ERRO ARQUIVO]    Não foi possível remover {file}: {e}")

    print("\n✨ Limpeza concluída com sucesso! Próxima execução pronta.")

if __name__ == "__main__":
    clean_directory()
