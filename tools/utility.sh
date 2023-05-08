directory=$(dirname $0)

count() {
  wc -l
}

filter_api() {
  grep -o '[mM]lir[A-Z][a-zA-Z0-9]*' | grep -iv -e python -e isnull | sort -u
}

implemented_api() {
  (
    cd $directory/../melior

    cargo install cargo-expand
    cargo expand | filter_api
  )
}

all_api() {
  cat $(find $(brew --prefix llvm)/include/mlir-c -type f) | filter_api
}
