#!/usr/bin/env bash
#
# Automacao dos testes do CGPGRN (CUDA).
#
# Este script fica NA PASTA DAS BASES, junto com os arquivos de entrada.
# Os arquivos sao pareados pela ordem do nome:
#
#   1o arquivo de expressao  + 1o arquivo de pseudotempo  + 1o COMANDO
#   2o arquivo de expressao  + 2o arquivo de pseudotempo  + 2o COMANDO
#   ...
#
#   - expressao  : .csv com "express" no nome (ex.: 01_ExpressionData_mCAD.csv)
#   - pseudotempo: .csv com "pseudo"  no nome (ex.: 01_PseudoTime_mCAD.csv)
#   Dica: numere os arquivos (01_, 02_, ...) para garantir a ordem.
#
# Para cada teste (repetido REPETICOES[i] vezes):
#   1. copia os dois arquivos para a pasta do framework como
#      ExpressionData.csv e PseudoTime.csv
#   2. roda o COMANDO dentro da pasta do framework e espera terminar
#   3. move progCUDA e Makefile para dentro da pasta finalResults_* gerada
#   4. move a pasta finalResults_* para RESULTADOS_GERAIS
#      (renomeada para finalResults_<pn>_rep2, _rep3 nas repeticoes)
#   5. o que mais sobrar na pasta do framework vai junto, em
#      finalResults_*/sobras_execucao (nada e apagado)
#
# Uso (com o venv ativado ou nao, o script ativa sozinho):
#   ./rodar_bases.sh          # mostra os pares e pede confirmacao
#   ./rodar_bases.sh -y       # comeca direto, sem perguntar

set -u
shopt -s nullglob

# =============================== CONFIGURACAO ===============================

# Pasta do framework (onde ficam CGPGRN.py, source/, kernels/ ...)
FRAMEWORK="$HOME/Downloads/tese-main/cgpgrn-CUDA/cgpgrn-CUDA"

# Para onde vao as pastas finalResults_* de cada teste
RESULTADOS_GERAIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/resultados_gerais"

# venv com pandas, sklearn etc.
VENV="$HOME/Downloads/tese-main/venv/bin/activate"

# Um comando por teste, NA MESMA ORDEM dos arquivos.
# Use sempre -e ExpressionData.csv -p PseudoTime.csv (o script copia com esses nomes).
COMANDOS=(
  'python CGPGRN.py -e ExpressionData.csv -p PseudoTime.csv -cm KMeans -nc 4 -pn mCAD -s 500cells -d BiKMeans -r 5 -run'  # 1 mCAD (5 genes)
  'python CGPGRN.py -e ExpressionData.csv -p PseudoTime.csv -cm KMeans -nc 8 -pn mESC -s 500cells -d BiKMeans -r 5 -run'  # 2 mESC (16 genes)
  'python CGPGRN.py -e ExpressionData.csv -p PseudoTime.csv -cm KMeans -nc 10 -pn GSD -s 500cells -d BiKMeans -r 5 -run'  # 3 GSD (19 genes)
  'python CGPGRN.py -e ExpressionData.csv -p PseudoTime.csv -cm KMeans -nc 16 -pn Ecoli-32 -s 500cells -d BiKMeans -r 5 -run'  # 4 Ecoli-32 (32 genes)
  'python CGPGRN.py -e ExpressionData.csv -p PseudoTime.csv -cm KMeans -nc 32 -pn Ecoli-64 -s 500cells -d BiKMeans -r 5 -run'  # 5 Ecoli-64 (64 genes)
  'python CGPGRN.py -e ExpressionData.csv -p PseudoTime.csv -cm KMeans -nc 32 -pn mDC -s 500cells -d BiKMeans -r 5 -run'  # 6 mDC (128 genes)
)

# Quantas vezes rodar cada teste, NA MESMA ORDEM de COMANDOS.
# Regra do professor: bases com ate 19 genes -> 3 vezes; as demais -> 1 vez.
# Pastas: 1a vez = finalResults_<pn>, 2a = finalResults_<pn>_rep2, 3a = finalResults_<pn>_rep3.
REPETICOES=(3 3 3 1 1 1)   # mCAD mESC GSD Ecoli-32 Ecoli-64 mDC

# ============================================================================

BASES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arquivos/pastas do proprio framework: nunca sao movidos.
MANTER=(CGPGRN.py clean.py include kernels source)

manter() {
    local m
    for m in "${MANTER[@]}"; do [ "$1" = "$m" ] && return 0; done
    case "$1" in cuda_*.run*|*.md) return 0 ;; esac
    return 1
}

# Move para $1 tudo da pasta do framework que nao e do framework.
guardar_sobras() {
    local destino="$1" item moveu=1
    for item in *; do
        manter "$item" && continue
        mkdir -p "$destino"
        mv -- "$item" "$destino/" && moveu=0
    done
    return $moveu
}

# Valor do -pn de um comando (nome da pasta finalResults_<pn>).
pn_de() { echo "$1" | grep -oE '(-pn|--problemName)[[:space:]]+[^[:space:]]+' | awk '{print $2}'; }

# Nome da pasta de resultado da repeticao $2 do teste de -pn $1.
pasta_rep() { if [ "$2" -eq 1 ]; then echo "finalResults_$1"; else echo "finalResults_$1_rep$2"; fi; }

# Verdadeiro se a repeticao $2 do comando $1 ja terminou antes com sucesso.
ja_concluido() {
    local pn
    pn="$(pn_de "$1")"
    [ -n "$pn" ] && [ -d "$RESULTADOS_GERAIS/$(pasta_rep "$pn" "$2")" ]
}

# Quantas repeticoes do comando $1 ja existem (de 1 ate $2).
reps_feitas() {
    local r n=0
    for ((r = 1; r <= $2; r++)); do ja_concluido "$1" "$r" && n=$((n + 1)); done
    echo "$n"
}

# Caminho livre em RESULTADOS_GERAIS (nunca sobrescreve um resultado antigo).
destino_livre() {
    local base="$RESULTADOS_GERAIS/$1" alvo i=2
    alvo="$base"
    while [ -e "$alvo" ]; do alvo="${base}_$i"; i=$((i + 1)); done
    echo "$alvo"
}

# ------------------------------ arquivos ------------------------------------
EXPRS=(); PSEUDOS=()
while IFS= read -r f; do
    n="$(basename "$f" | tr '[:upper:]' '[:lower:]')"
    case "$n" in
        *pseudo*)  PSEUDOS+=("$f") ;;
        *express*) EXPRS+=("$f") ;;
    esac
done < <(find "$BASES" -maxdepth 1 -type f -iname '*.csv' -printf '%f\n' | sort -V | sed "s|^|$BASES/|")

NE=${#EXPRS[@]}; NP=${#PSEUDOS[@]}; NC=${#COMANDOS[@]}; NR=${#REPETICOES[@]}

echo "Pasta das bases : $BASES"
echo "Framework       : $FRAMEWORK"
echo "Resultados      : $RESULTADOS_GERAIS"
echo
for ((i = 0; i < NE || i < NP || i < NC; i++)); do
    echo "Teste $((i + 1)):"
    echo "  expressao  : $(basename "${EXPRS[i]:-<FALTANDO>}")"
    echo "  pseudotempo: $(basename "${PSEUDOS[i]:-<FALTANDO>}")"
    echo "  comando    : ${COMANDOS[i]:-<FALTANDO>}"
    if [ -n "${COMANDOS[i]:-}" ] && [ -n "${REPETICOES[i]:-}" ]; then
        feitas="$(reps_feitas "${COMANDOS[i]}" "${REPETICOES[i]}")"
        faltam=$(( REPETICOES[i] - feitas ))
        echo "  repeticoes : ${REPETICOES[i]}  (ja feitas: $feitas, faltam: $faltam)"
        [ "$faltam" -eq 0 ] && echo "  >> ja concluido: sera pulado"
    fi
done
echo

if [ "$NE" -eq 0 ] || [ "$NE" -ne "$NP" ] || [ "$NE" -ne "$NC" ] || [ "$NE" -ne "$NR" ]; then
    echo "ERRO: as quantidades nao batem: $NE expressao, $NP pseudotempo, $NC comandos, $NR repeticoes." >&2
    echo "Ajuste os arquivos ou as listas COMANDOS/REPETICOES no topo do script." >&2
    exit 1
fi

if [ "${1:-}" != "-y" ]; then
    read -r -p "Os pares acima estao certos? Rodar os $NE testes? [s/N] " resp
    case "$resp" in s|S|sim|SIM) ;; *) echo "Cancelado."; exit 0 ;; esac
fi

# ------------------------------ ambiente ------------------------------------
if [ ! -f "$FRAMEWORK/CGPGRN.py" ]; then
    echo "ERRO: CGPGRN.py nao encontrado em $FRAMEWORK (ajuste FRAMEWORK no topo)." >&2
    exit 1
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV" ]; then
    # shellcheck disable=SC1090
    source "$VENV"
fi
export PATH="/usr/local/cuda/bin:$PATH"
if ! python -c "import pandas, sklearn, scipy, csaps" 2>/dev/null; then
    echo "ERRO: o python nao tem as dependencias. Ative o venv:  source $VENV" >&2
    exit 1
fi
if ! command -v nvcc > /dev/null; then
    echo "ERRO: nvcc nao encontrado (esperado em /usr/local/cuda/bin)." >&2
    exit 1
fi

mkdir -p "$RESULTADOS_GERAIS"
RESUMO="$RESULTADOS_GERAIS/resumo.tsv"
[ -f "$RESUMO" ] || printf "inicio\tteste\tstatus\tduracao\texpressao\tpseudotempo\tpasta\tcomando\n" > "$RESUMO"

cd "$FRAMEWORK" || exit 1

# Sobras de uma execucao anterior (ex.: interrompida) nao se misturam com o 1o teste.
SOBRAS_ANT="$(destino_livre "sobras_anteriores_$(date +%Y%m%d_%H%M%S)")"
if guardar_sobras "$SOBRAS_ANT"; then
    echo "Arquivos de uma execucao anterior foram movidos para: $SOBRAS_ANT"
fi

# ----------------------------- interrupcao ----------------------------------
ATUAL=""; REP=1; TMP_LOG=""; INICIO_TXT=""; CMD=""; EXPR=""; PSEUDO=""
interrompido() {
    trap - INT TERM
    echo
    echo "Interrompido. Guardando o que foi gerado ate agora..."
    if [ -n "$ATUAL" ]; then
        local dest
        dest="$(destino_livre "INTERROMPIDO_teste${ATUAL}_rep${REP}_$(date +%Y%m%d_%H%M%S)")"
        guardar_sobras "$dest"
        mkdir -p "$dest"
        [ -f "$TMP_LOG" ] && mv -- "$TMP_LOG" "$dest/execucao.log"
        printf "%s\t%s\tINTERROMPIDO\t-\t%s\t%s\t%s\t%s\n" "$INICIO_TXT" "$ATUAL rep$REP" \
            "$(basename "$EXPR")" "$(basename "$PSEUDO")" "$dest" "$CMD" >> "$RESUMO"
        echo "Salvo em: $dest"
    fi
    exit 130
}
trap interrompido INT TERM

# ------------------------------- testes -------------------------------------
for ((i = 0; i < NE; i++)); do
    ATUAL=$((i + 1))
    EXPR="${EXPRS[i]}"; PSEUDO="${PSEUDOS[i]}"; CMD="${COMANDOS[i]}"
    NREP="${REPETICOES[i]}"; PN="$(pn_de "$CMD")"
  for ((REP = 1; REP <= NREP; REP++)); do
    INICIO_TXT="$(date '+%F %T')"

    # Repeticao ja concluida antes (existe resultados_gerais/<pasta da repeticao>): pula.
    if ja_concluido "$CMD" "$REP"; then
        echo "[$ATUAL/$NE] $PN rep $REP/$NREP ja concluida (resultados_gerais/$(pasta_rep "$PN" "$REP")). Pulando."
        continue
    fi

    echo "=================================================================="
    echo "[$ATUAL/$NE] $PN - repeticao $REP de $NREP - $INICIO_TXT"
    echo "  expressao  : $(basename "$EXPR")"
    echo "  pseudotempo: $(basename "$PSEUDO")"
    echo "  comando    : $CMD"
    echo "=================================================================="

    # 1. entradas
    cp -- "$EXPR" ExpressionData.csv
    cp -- "$PSEUDO" PseudoTime.csv

    # 2. execucao (o log fica fora da pasta do framework ate o fim)
    TMP_LOG="$RESULTADOS_GERAIS/.execucao_teste${ATUAL}_rep${REP}.log"
    T0=$(date +%s)
    bash -c "$CMD" < /dev/null 2>&1 | tee "$TMP_LOG"
    DUR=$(( $(date +%s) - T0 ))
    DUR_TXT="$(printf '%02dh%02dm%02ds' $((DUR/3600)) $((DUR%3600/60)) $((DUR%60)))"

    # 3-5. organizar resultados
    FINAIS=(finalResults_*/)
    if [ ${#FINAIS[@]} -gt 0 ]; then
        STATUS="OK"
        FINAL="${FINAIS[0]%/}"
        [ -e progCUDA ] && mv -- progCUDA "$FINAL/"
        [ -e Makefile ] && mv -- Makefile "$FINAL/"
        DEST="$(destino_livre "$(pasta_rep "$PN" "$REP")")"
        mv -- "$FINAL" "$DEST"
        guardar_sobras "$DEST/sobras_execucao"
    else
        STATUS="FALHOU"
        DEST="$(destino_livre "FALHOU_teste${ATUAL}_rep${REP}_$(date +%Y%m%d_%H%M%S)")"
        guardar_sobras "$DEST"
    fi
    mkdir -p "$DEST"
    mv -- "$TMP_LOG" "$DEST/execucao.log"
    {
        echo "teste: $ATUAL de $NE"
        echo "repeticao: $REP de $NREP"
        echo "expressao: $EXPR"
        echo "pseudotempo: $PSEUDO"
        echo "comando: $CMD"
        echo "inicio: $INICIO_TXT"
        echo "duracao: $DUR_TXT"
        echo "status: $STATUS"
    } > "$DEST/teste.txt"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$INICIO_TXT" "$ATUAL rep$REP" "$STATUS" "$DUR_TXT" \
        "$(basename "$EXPR")" "$(basename "$PSEUDO")" "$DEST" "$CMD" >> "$RESUMO"

    echo ">> Teste $ATUAL ($PN) repeticao $REP/$NREP: $STATUS em $DUR_TXT -> $DEST"
  done
    ATUAL=""
done

echo
echo "Fim. Resumo: $RESUMO"
column -t -s $'\t' "$RESUMO" 2>/dev/null || cat "$RESUMO"
