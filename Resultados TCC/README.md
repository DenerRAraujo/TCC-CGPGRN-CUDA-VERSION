# Resultados do TCC — CGPGRN em OpenCL × CUDA

Esta pasta reúne todos os experimentos usados no TCC *Inferência de Redes de
Regulação Gênica com Programação Genética Cartesiana em CUDA*. Há uma subpasta
para cada versão do CGPGRN, com a mesma organização:

```
Resultados TCC/
├── OpenCL/                         versão original (inferência em OpenCL)
│   ├── 01_mCAD_ExpressionData.csv  dados de entrada (iguais nas duas versões)
│   ├── 01_mCAD_PseudoTime.csv
│   ├── ...                         02_mESC, 03_GSD, 04_Ecoli-32, 05_Ecoli-64, 06_mDC
│   ├── rodar_bases_opencl.sh       script que executou todos os testes
│   ├── rodar.log                   saída completa da execução
│   ├── resultados_gerais/          RESULTADOS USADOS NO TCC
│   └── resultados_com_trava/       execuções extras (não usadas no TCC)
└── CUDA/                           versão nova (inferência em CUDA)
    ├── (mesmos arquivos de entrada)
    ├── rodar_bases.sh
    ├── rodar.log
    └── resultados_gerais/          RESULTADOS USADOS NO TCC
```

## Problemas e repetições

| Problema | Genes | Tipo (origem) | Repetições por versão |
|---|---|---|---|
| mCAD | 5 | Curado (BEELINE) | 3 |
| mESC | 16 | Experimental (BEELINE) | 3 |
| GSD | 19 | Curado (BEELINE) | 3 |
| Ecoli-32 | 32 | Sintético (DREAM4) | 1 |
| Ecoli-64 | 64 | Sintético (DREAM4) | 1 |
| mDC | 128 | Experimental (BEELINE) | 1 |

Cada repetição está em uma pasta própria dentro de `resultados_gerais/`:

- 1ª repetição: `finalResults_<problema>`
- 2ª e 3ª repetições: `finalResults_<problema>_rep2` e `finalResults_<problema>_rep3`

## Como chegar à rede inferida (`rankedEdges`)

O caminho segue sempre o mesmo padrão:

```
Resultados TCC/<versão>/resultados_gerais/finalResults_<problema>[_repN]/rankedEdges_<problema>.csv
```

Exemplos:

| O que | Caminho |
|---|---|
| mCAD, OpenCL, 1ª repetição | `OpenCL/resultados_gerais/finalResults_mCAD/rankedEdges_mCAD.csv` |
| mCAD, CUDA, 3ª repetição | `CUDA/resultados_gerais/finalResults_mCAD_rep3/rankedEdges_mCAD.csv` |
| GSD, OpenCL, 2ª repetição | `OpenCL/resultados_gerais/finalResults_GSD_rep2/rankedEdges_GSD.csv` |
| Ecoli-64, CUDA | `CUDA/resultados_gerais/finalResults_Ecoli-64/rankedEdges_Ecoli-64.csv` |
| mDC, OpenCL | `OpenCL/resultados_gerais/finalResults_mDC/rankedEdges_mDC.csv` |

### Os níveis de `rankedEdges` dentro de cada pasta

Cada pasta `finalResults_*` contém arquivos de `rankedEdges` em três níveis.
Do mais consolidado para o mais detalhado:

Antes, uma nota sobre os termos. Cada teste foi rodado com a opção `-r 5`, que
executa o programa de inferência 5 vezes, em **rodadas** independentes
(`exe_1` a `exe_5`). Dentro de cada rodada, a CGP é executada $n_g = 5$ vezes
em paralelo na GPU para cada gene alvo.

1. **Rede final:** `rankedEdges_<problema>.csv`, na raiz da pasta.
   Junta as 5 rodadas e todos os grupos de genes em uma única lista, sem
   arestas repetidas e ordenada pelo peso. Quando uma aresta aparece em mais
   de uma rodada, fica o maior peso. **Foi esse o arquivo comparado no TCC**
   (Tabela de similaridade).
2. **Rede de cada rodada:** `rankedEdges_<problema>_exe_<k>.csv`, na raiz da
   pasta, com `k` de 1 a 5. É a rede consolidada usando apenas a rodada `k`.
3. **Redes parciais:** `executions_parallel/exe_<k>_Cluster<c>_pt<p>/rankedEdges_<problema>.csv`.
   Saída bruta da rodada `k` para o grupo de genes `c`, antes da
   consolidação. Ao lado ficam um `.txt` por gene alvo, com o circuito
   evoluído para aquele gene, e o `unfeasibles_<problema>.txt`.

Os arquivos `unified_rankedEdges_*.csv` são etapas intermediárias: são a
simples concatenação das redes parciais, com arestas repetidas, que o
pós-processamento usa para gerar os arquivos dos níveis 1 e 2.

### Formato do arquivo

Texto separado por tabulação, uma aresta por linha:

```
Gene1   Gene2   EdgeWeight
Fgf8    Pax6    1.0
Sp8     Pax6    0.8
```

`Gene1` regula `Gene2`, e `EdgeWeight` é a probabilidade da regulação: a
fração das $n_g = 5$ execuções paralelas da CGP em que `Gene1` apareceu no
circuito de `Gene2` (0,8 = 4 de 5). As redes parciais do nível 3 não têm a
linha de cabeçalho.

## Outros arquivos úteis em cada `finalResults_*`

| Arquivo | Conteúdo |
|---|---|
| `timeStatistics_<problema>.txt` | Tempos de execução. O TCC usa o total de `timeIterTotal`. |
| `CGPGRN_parameters.txt` | Parâmetros e grupos de genes da execução. |
| `teste.txt` | Comando executado, arquivos de entrada, início, duração e status. |
| `Makefile` | Número de nós (`MAXNODESVALUE`) e de gerações (`MAXEVAL`) da CGP. |
| `discretizedData/` | Dados discretizados (idênticos em todas as execuções de um problema). |
| `kmeans/Cluster<c>/` | Genes de cada grupo (`geneNames_*.txt`) e tabelas-verdade. |
| `execucao.log`, `LogFile_<problema>.txt` | Logs da execução. |

O arquivo `resultados_gerais/resumo.tsv` lista todas as execuções da versão,
com data, duração, status e comando usado.

## Pastas que não entram no TCC

| Pasta | Motivo |
|---|---|
| `OpenCL/resultados_com_trava/` | Execuções extras do mESC e do mDC, fora da série comparada. |
| `OpenCL/resultados_gerais/finalResults_mCAD_OpenCL/` | Execução antiga do mCAD, anterior à série de repetições. |
| `OpenCL/resultados_gerais/sobras_anteriores_*/` | Restos de uma execução anterior do mDC. |
| `CUDA/resultados_gerais/INTERROMPIDO_teste4_*/` | Execução do Ecoli-32 interrompida e refeita depois. |

Observação: a 1ª repetição do GSD em OpenCL foi executada com o nome de
problema `CSD`, por erro de digitação no comando. Os arquivos e o conteúdo
foram renomeados para `GSD`. Os dados de entrada e a discretização são os
mesmos das demais repetições do GSD.

## Comparando as duas versões

O script `../Avaliacao/comparar_redes.py` compara automaticamente os arquivos
`rankedEdges_<problema>.csv` (nível 1) das duas versões e gera as tabelas do
TCC em `../Avaliacao/resultados/`. Na raiz do repositório:

```bash
python Avaliacao/comparar_redes.py
```
