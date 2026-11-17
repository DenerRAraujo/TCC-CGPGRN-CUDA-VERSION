"""
Metricas de avaliacao do BEELINE (Pratapa et al., 2020) para as redes inferidas
pelo CGPGRN nas versoes OpenCL e CUDA.

Avalia apenas os problemas executados com a base de expressao COMPLETA, ou seja,
cujo arquivo de entrada e identico ao do problema original em Problemas/ e que
possuem rede de referencia propria:

    mCAD     -> Problemas/Curated/mCAD/mCAD-2000-1/refNetwork.csv
    GSD      -> Problemas/Curated/GSD/GSD-2000-1/refNetwork.csv
    Ecoli-32 -> Problemas/E. Coli/32/Ecoli-1-32-refNetwork.csv
    Ecoli-64 -> Problemas/E. Coli/64/Ecoli-1-64-refNetwork.csv

(mESC e mDC foram executados com subconjuntos de genes das bases experimentais
e por isso nao entram aqui.)

Calculo, como no BEELINE (rede direcionada, sem lacos):
  - arestas candidatas: todos os pares ordenados (a, b) de genes, a != b;
  - arestas nao previstas recebem peso 0;
  - AUPRC e AUROC: area sob as curvas precisao-revocacao e ROC (sklearn);
  - EP: entre as k arestas de maior peso (k = arestas da referencia), incluindo
    empates com a k-esima, fracao que esta na referencia;
  - AUPRC ratio e EPR: valores divididos pelo de um preditor aleatorio
    (densidade da rede de referencia).

Saidas em Avaliacao/resultados/:
  beeline_por_execucao.csv   uma linha por rede final (versao x repeticao)
  beeline_resumo.csv         media e desvio padrao por problema e versao
  tabela_beeline.tex         tabela pronta para o TCC

Uso (na raiz do repositorio):
  python Avaliacao/metricas_beeline.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "Avaliacao" / "resultados"
RESULTADOS = RAIZ / "Resultados TCC"
VERSOES = ["OpenCL", "CUDA"]

P = RAIZ / "Problemas"
PROBLEMAS = {
    "mCAD": ("01_mCAD_ExpressionData.csv", P / "Curated/mCAD/mCAD-2000-1/refNetwork.csv"),
    "GSD": ("03_GSD_ExpressionData.csv", P / "Curated/GSD/GSD-2000-1/refNetwork.csv"),
    "Ecoli-32": ("04_Ecoli-32_ExpressionData.csv", P / "E. Coli/32/Ecoli-1-32-refNetwork.csv"),
    "Ecoli-64": ("05_Ecoli-64_ExpressionData.csv", P / "E. Coli/64/Ecoli-1-64-refNetwork.csv"),
}
PASTA_RE = re.compile(r"^finalResults_(?P<prob>[^_]+)(?:_rep(?P<rep>\d+))?$")


def genes_do_problema(arquivo):
    expr = pd.read_csv(RESULTADOS / "CUDA" / arquivo, index_col=0)
    return sorted(map(str, expr.index))


def carregar_referencia(arquivo, genes):
    ref = pd.read_csv(arquivo)
    g = set(genes)
    return {(a, b) for a, b in zip(ref.iloc[:, 0].astype(str), ref.iloc[:, 1].astype(str))
            if a != b and a in g and b in g}


def carregar_predicao(arquivo):
    df = pd.read_csv(arquivo, sep="\t")
    pred = {}
    for a, b, w in zip(df.Gene1.astype(str), df.Gene2.astype(str), df.EdgeWeight.astype(float)):
        if a != b and w > 0:
            pred[(a, b)] = max(abs(w), pred.get((a, b), 0.0))
    return pred


def metricas(pred, ref, genes):
    candidatas = [(a, b) for a in genes for b in genes if a != b]
    y = np.array([e in ref for e in candidatas], dtype=int)
    s = np.array([pred.get(e, 0.0) for e in candidatas])
    densidade = y.mean()

    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    auprc, auroc = auc(rec, prec), auc(fpr, tpr)

    pesos = sorted(pred.values(), reverse=True)
    k = min(len(pesos), len(ref))
    if k == 0:
        ep = 0.0
    else:
        corte = pesos[k - 1]
        topo = [e for e, w in pred.items() if w >= corte]
        ep = sum(e in ref for e in topo) / len(topo)

    return dict(arestas_referencia=len(ref), arestas_previstas=len(pred),
                AUROC=auroc, AUPRC=auprc, AUPRC_ratio=auprc / densidade,
                EP=ep, EPR=ep / densidade)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = []
    for prob, (arq_expr, arq_ref) in PROBLEMAS.items():
        genes = genes_do_problema(arq_expr)
        ref = carregar_referencia(arq_ref, genes)
        for versao in VERSOES:
            base = RESULTADOS / versao / "resultados_gerais"
            for pasta in sorted(base.iterdir()):
                m = PASTA_RE.match(pasta.name)
                if not (pasta.is_dir() and m and m["prob"] == prob):
                    continue
                arq = pasta / f"rankedEdges_{prob}.csv"
                if not arq.exists():
                    continue
                linhas.append(dict(problema=prob, genes=len(genes), versao=versao,
                                   repeticao=int(m["rep"] or 1),
                                   arquivo=str(arq.relative_to(RAIZ)),
                                   **metricas(carregar_predicao(arq), ref, genes)))

    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "beeline_por_execucao.csv", index=False)

    cols = ["AUROC", "AUPRC", "AUPRC_ratio", "EP", "EPR"]
    resumo = df.groupby(["problema", "genes", "versao"], sort=False)[cols].agg(["mean", "std"])
    resumo.columns = [f"{c}_{s}" for c, s in resumo.columns]
    resumo.insert(0, "repeticoes", df.groupby(["problema", "genes", "versao"], sort=False).size())
    resumo = resumo.reset_index()
    resumo.to_csv(SAIDA / "beeline_resumo.csv", index=False)
    escrever_tabela(df)

    pd.set_option("display.width", 200)
    print(df[["problema", "versao", "repeticao", "arestas_referencia", "arestas_previstas",
              "AUROC", "AUPRC", "AUPRC_ratio", "EP", "EPR"]].round(3).to_string(index=False))
    print("\n=== Media (desvio) por problema e versao ===")
    print(resumo.round(3).to_string(index=False))
    print(f"\nArquivos gravados em {SAIDA.relative_to(RAIZ)}")


def escrever_tabela(df):
    def fmt(s):
        txt = f"{s.mean():.3f}" if len(s) == 1 else f"{s.mean():.3f} $\\pm$ {s.std():.3f}"
        return txt.replace(".", ",")

    linhas = []
    for prob in PROBLEMAS:
        for versao in VERSOES:
            g = df[(df.problema == prob) & (df.versao == versao)]
            if g.empty:
                continue
            primeira = prob if versao == VERSOES[0] else ""
            linhas.append(" & ".join([primeira, versao, str(len(g)),
                                      fmt(g.AUROC), fmt(g.AUPRC), fmt(g.EP)]) + r" \\")
        linhas.append(r"\hline")

    tex = r"""\begin{table}[htb]
    \centering
    \caption{Métricas do BEELINE para as redes inferidas pelas versões OpenCL
    e CUDA nos problemas com base de expressão completa. Com mais de uma
    repetição, os valores são média $\pm$ desvio padrão amostral.}
    \label{tab:beeline}
    \begin{tabular}{l l c c c c}
        \hline
        Problema & Versão & Rep. & AUROC & AUPRC & EP \\
        \hline
""" + "\n".join("        " + l for l in linhas) + r"""
    \end{tabular}
\end{table}
"""
    (SAIDA / "tabela_beeline.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
