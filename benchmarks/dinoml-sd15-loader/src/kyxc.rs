use model_weights::Result;
use model_weights::identity::StableName;
use model_weights::prepare::Layout;

const LAYOUT_NAME: &str = "dinoml.ck_conv2d_weight.kyxc";
const LAYOUT_VERSION: u32 = 1;
const LAYOUT_PARAMETERS: &[u8] = b"oihw-to-kyxc/no-padding";

/// Returns the exact `DinoML` CK convolution-weight storage ABI.
pub fn layout() -> Result<Layout> {
    Ok(Layout::custom(
        StableName::parse(LAYOUT_NAME)?,
        LAYOUT_VERSION,
        LAYOUT_PARAMETERS,
    ))
}
