"""
Compara as redes inferidas pelo CGPGRN em OpenCL e em CUDA.

Para cada problema, compara o arquivo final rankedEdges_<problema>.csv de cada
repeticao OpenCL com o de cada repeticao CUDA. Como linha de base, compara
tambem repeticoes da mesma versao entre si (variacao natural do framework).

Medidas por par de CSVs:
  identico          os dois arquivos tem exatamente as mesmas arestas e pesos
  jaccard           |arestas em comum| / |arestas na uniao| (1 = mesmas arestas)
  mesmo_peso        fracao das arestas em comum com o mesmo peso
  mesmo_agrupamento os dois rodaram com os mesmos grupos de genes (K-means)

Saidas em Avaliacao/resultados/:
  similaridade_pares.csv     uma linha por par comparado
  similaridade_resumo.csv    media por problema e tipo de comparacao
  tabela_similaridade.tex    tabela pronta para o TCC

Uso (a partir da raiz do repositorio):
  python Avaliacao/avaliar_qualidade.py
"""

import itertools
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "Avaliacao" / "resultados"

VERSOES = {
    "OpenCL": RAIZ / "Resultados TCC" / "OpenCL" / "resultados_gerais",
    "CUDA": RAIZ / "Resultados TCC" / "CUDA" / "resultados_gerais",
}
PROBLEMAS = ["mCAD", "mESC", "GSD", "Ecoli-32", "Ecoli-64", "mDC"]
PASTA_RE = re.compile(r"^finalResults_(?P<prob>[^_]+)(?:_rep(?P<rep>\d+))?$")


def carregar_rede(arquivo):
    """{(regulador, alvo): peso} a partir de um rankedEdges."""
    df = pd.read_csv(arquivo, sep="\t")
    return {(a, b): round(float(w), 6)
            for a, b, w in zip(df.Gene1.astype(str), df.Gene2.astype(str), df.EdgeWeight)}


def carregar_grupos(pasta):
    """Conjunto de grupos de genes usados na execucao (arquivos geneNames)."""
    grupos = set()
    for f in (pasta / "kmeans").glob("Cluster*/geneNames_*.txt"):
        grupos.add(frozenset(f.read_text().split()))
    return frozenset(grupos)


def listar_execucoes():
    execs = {}
    for versao, base in VERSOES.items():
        for pasta in sorted(base.iterdir()):
            m = PASTA_RE.match(pasta.name)
            if not (pasta.is_dir() and m and m["prob"] in PROBLEMAS):
                continue
            arq = pasta / f"rankedEdges_{m['prob']}.csv"
            if not arq.exists():  # ex.: finalResults_GSD do OpenCL contem o CSD
                print(f"  ignorando {versao}/{pasta.name}: sem rankedEdges_{m['prob']}.csv")
                continue
            execs[(m["prob"], versao, int(m["rep"] or 1))] = (carregar_rede(arq), carregar_grupos(pasta))
    return execs


def comparar(a, b):
    ra, rb = a[0], b[0]
    comuns = ra.keys() & rb.keys()
    uniao = ra.keys() | rb.keys()
    return dict(
        arestas_a=len(ra), arestas_b=len(rb), arestas_comuns=len(comuns),
        jaccard=len(comuns) / len(uniao) if uniao else 1.0,
        mesmo_peso=sum(ra[e] == rb[e] for e in comuns) / len(comuns) if comuns else 0.0,
        identico=ra == rb,
        mesmo_agrupamento=a[1] == b[1],
    )


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    execs = listar_execucoes()

    pares = []
    for prob in PROBLEMAS:
        chaves = sorted(k for k in execs if k[0] == prob)
        for ka, kb in itertools.combinations(chaves, 2):
            tipo = "OpenCL x CUDA" if ka[1] != kb[1] else f"{ka[1]} x {kb[1]}"
            pares.append(dict(problema=prob, comparacao=tipo,
                              rede_a=f"{ka[1]} rep{ka[2]}", rede_b=f"{kb[1]} rep{kb[2]}",
                              **comparar(execs[ka], execs[kb])))
    df = pd.DataFrame(pares)
    df.to_csv(SAIDA / "similaridade_pares.csv", index=False)

    resumo = (df.groupby(["problema", "comparacao"], sort=False)
                .agg(pares=("jaccard", "size"), jaccard=("jaccard", "mean"),
                     mesmo_peso=("mesmo_peso", "mean"), identicos=("identico", "sum"),
                     mesmo_agrupamento=("mesmo_agrupamento", "sum"))
                .reset_index())
    resumo.to_csv(SAIDA / "similaridade_resumo.csv", index=False)
    escrever_tabela(df)

    pd.set_option("display.width", 200)
    print(df[["problema", "rede_a", "rede_b", "arestas_a", "arestas_b", "jaccard",
              "mesmo_peso", "identico", "mesmo_agrupamento"]].round(3).to_string(index=False))
    print("\n=== Resumo ===")
    print(resumo.round(3).to_string(index=False))
    print(f"\nArquivos gravados em {SAIDA.relative_to(RAIZ)}")


def escrever_tabela(df):
    """Uma linha por problema: OpenCL x CUDA e, como base, a mesma versao."""
    linhas = []
    for prob in PROBLEMAS:
        g = df[df.problema == prob]
        if g.empty:
            continue
        x = g[g.comparacao == "OpenCL x CUDA"]
        base = g[g.comparacao != "OpenCL x CUDA"]
        fmt = lambda v: f"{v:.2f}".replace(".", ",")
        linhas.append(" & ".join([
            prob,
            f"{int(x.identico.sum())}/{len(x)}",
            fmt(x.jaccard.mean()), fmt(x.mesmo_peso.mean()),
            fmt(base.jaccard.mean()) if len(base) else "--",
        ]) + r" \\")
    tex = r"""\begin{table}[htb]
    \centering
    \caption{Semelhança entre as redes inferidas pelas versões OpenCL e CUDA.}
    \label{tab:similaridade}
    \begin{tabular}{l c c c c}
        \hline
        & \multicolumn{3}{c}{OpenCL $\times$ CUDA} & Mesma versão \\
        Problema & Idênticas & Jaccard & Mesmo peso & Jaccard \\
        \hline
""" + "\n".join("        " + l for l in linhas) + r"""
        \hline
    \end{tabular}
\end{table}
"""
    (SAIDA / "tabela_similaridade.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
