# https://pypi.org/project/pdftotext/
import re, os, glob, pdftotext

# Load your PDF
data_rout = r"./data/tax_dem_exampls_pdf"
result_rout = r"./data/tax_dems_txt"
for fn in glob.glob(os.path.join(data_rout, "*.pdf")):
    with open(fn, "rb") as f:
        pdf = pdftotext.PDF(f)
    tx = "\n\n".join(pdf)
    fn_cut = re.sub(data_rout+"/|.pdf", "", fn)
    # print(fn_cut)
    txt_fn = fn_cut + ".txt"
    with open(os.path.join(result_rout, txt_fn), "w") as f:
        f.write(tx)