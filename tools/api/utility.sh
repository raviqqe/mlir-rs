count() {
  wc -l
}

filter_api() {
  grep -o '[mM]lir[A-Z][a-zA-Z0-9]*' | grep -iv -e python -e isnull | sort -u
}

implemented_api() {
  (
    cd $(dirname $0)/../../melior

    cargo install cargo-expand
    cargo expand | filter_api
  )
}

all_api() {
  cat $(find $(brew --prefix llvm)/include -type f) | filter_api
}
