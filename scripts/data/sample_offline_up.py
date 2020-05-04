import numpy as np
import os

data_home_dir = "/home/t-yirwan/data/data/wmt"

PAIRS_YEAR = {'deen': 19, 'fien': 19, 'csen': 19, 'guen': 19, 'eten': 18, 'tren': 18, 'lven': 17, 'roen': 16,
              'fren': 15, 'hien': 14}
"""
# l07d01
PAIRS_TEMP2 = {'deen': 4613192, 'eten': 1793657, 'fien': 3957011, 'lven': 2578962, 'roen': 1580822, 'csen': 2002723, 'tren': 916716}
PAIRS_TEMP5 = {'deen': 4613192, 'lven': 3659403, 'tren': 2414620, 'eten': 3160215, 'csen': 3301933, 'fien': 4339507, 'roen': 3007433}
PAIRS_TEMP10 = {'deen': 4613192, 'roen': 3726365, 'lven': 4106756, 'csen': 3903584, 'fien': 4476391, 'tren': 3340431, 'eten': 3818268}
"""
PAIRS = {'gu': 85688, 'hi': 264199, 'tr': 182269, 'ro': 540562, 'et': 695227, 'lv': 1444235, 'cs': 10274582,
         'fi': 4844249, 'de': 4613192, 'fr': 10000000}
PAIRS_TEMP2 = None
PAIRS_TEMP5 = {'gu': 3944531, 'hi': 4940803, 'tr': 4587266, 'ro': 5701386, 'et': 5995655, 'lv': 6939671, 'cs': 10274582,
               'fi': 8840089, 'de': 8754103, 'fr': 10219069}
TEMP_DICT = {1: PAIRS, 2: PAIRS_TEMP2, 5: PAIRS_TEMP5}


def upsample_by_temp(temp, langs, tgt, setting_name, task="wmt_mtl"):
    print("\n" + "=" * 50)
    print("upsampling data with temp = {}".format(temp))

    text_dir = "{data_home_dir}/{task}/lib/text_{setting_name}".format(data_home_dir=data_home_dir, task=task,
                                                                       setting_name=setting_name)
    text_sample_dir = "{text_dir}_t{temp}".format(text_dir=text_dir, temp=temp)
    os.system("rm -r {text_sample_dir}/*".format(text_sample_dir=text_sample_dir))
    os.makedirs(text_sample_dir, exist_ok=True)
    pairs_temp = TEMP_DICT[temp]

    for src in langs:
        num = pairs_temp[src]
        settings = {
            "data_home_dir": data_home_dir,
            "text_dir": text_dir,
            "text_sample_dir": text_sample_dir,
            "src": src,
            "tgt": tgt,
            "temp": temp,
            "num": num
        }
        print("sampling {src}-{tgt} with {num}".format(**settings))

        fsrc = "{text_dir}/train_spm_{src}{tgt}.{src}".format(**settings)
        ftgt = "{text_dir}/train_spm_{src}{tgt}.{tgt}".format(**settings)
        fsrc_sp = "{text_sample_dir}/train_spm_{src}{tgt}.{src}".format(**settings)
        ftgt_sp = "{text_sample_dir}/train_spm_{src}{tgt}.{tgt}".format(**settings)
        fid_sp = "{text_sample_dir}/train_spm_{src}{tgt}.spidx".format(**settings)

        with open(fsrc, "r", encoding="utf8") as f:
            srclines = f.readlines()
        with open(ftgt, "r", encoding="utf8") as f:
            tgtlines = f.readlines()

        assert len(srclines) == len(tgtlines)
        n_lines = len(srclines)
        n_file = num // n_lines
        n_rem = num % n_lines
        print("  | total {} lines: target={}, n_file={}, n_rem={}".format(n_lines, num, n_file, n_rem))

        if n_rem > 0:
            print("  | randomly sample {} from {}".format(n_rem, n_lines))
            sidx = np.random.permutation(n_lines)
            topk = n_rem

            with open(fid_sp, "w", encoding="utf8") as f:
                for s in sidx[:topk]:
                    print(s, file=f)

            with open(fsrc_sp, "w", encoding="utf8") as f:
                for s in sidx[:topk]:
                    f.write(srclines[s])

            with open(ftgt_sp, "w", encoding="utf8") as f:
                for s in sidx[:topk]:
                    f.write(tgtlines[s])

        print("  | concat file {} times".format(n_file))
        for i in range(n_file):
            os.system("cat {fsrc} >> {fsrc_sp}".format(fsrc=fsrc, fsrc_sp=fsrc_sp))
            os.system("cat {ftgt} >> {ftgt_sp}".format(ftgt=ftgt, ftgt_sp=ftgt_sp))

    print("-" * 50)
    os.system("wc -l {text_sample_dir}/*".format(text_sample_dir=text_sample_dir))


if __name__ == "__main__":
    langs = ["de", "fi", "cs", "fr", "et", "tr", "lv", "ro", "hi", "gu"]
    # upsample_by_temp(temp=2, langs=langs, tgt="en")
    upsample_by_temp(temp=5, langs=langs, tgt="en", setting_name="l10d02")



