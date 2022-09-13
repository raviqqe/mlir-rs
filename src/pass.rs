use mlir_sys::{
    mlirCreateConversionConvertArithmeticToLLVM, mlirCreateConversionConvertFuncToLLVM, MlirPass,
};

pub struct Pass {
    raw: MlirPass,
}

impl Pass {
    pub fn convert_arithmetic_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertArithmeticToLLVM)
    }

    pub fn convert_func_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertFuncToLLVM)
    }

    fn from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self {
            raw: unsafe { create_raw() },
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirPass {
        self.raw
    }
}
