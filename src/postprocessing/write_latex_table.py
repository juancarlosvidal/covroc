# Generic LaTeX table writers used to build the simulation-scenario MSE tables
# (statistics_summary.py output) reported in the paper's Supplementary Material.
import os

class LatexTableWriter:
    """
    Class to create and append rows to a LaTeX table (.tex) with an arbitrary number of columns.
    """
    def __init__(self, filepath, column_headers, caption=None, label=None):
        if not column_headers or not isinstance(column_headers, (list, tuple)):
            raise ValueError("column_headers must be a non-empty list or tuple of header names.")
        self.filepath = filepath
        self.headers = column_headers
        self.caption = caption
        self.label = label
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        # Write header if new file
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("  \\centering\n")
                # Column format
                col_fmt = ' | '.join(['l'] * len(self.headers))
                f.write(f"  \\begin{{tabular}}{{| {col_fmt} |}}\n")
                f.write("    \\hline\n")
                f.write("    " + " & ".join(self.headers) + " \\\\ \n")
                f.write("    \\hline\n")

    def append_row(self, row):
        """Append one row matching the number of headers."""
        if len(row) != len(self.headers):
            raise ValueError(f"Row has {len(row)} entries but expected {len(self.headers)}.")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("    " + " & ".join(map(str, row)) + " \\\\ \n")
            f.write("    \\hline\n")

    def finalize(self):
        """Close the tabular and table environments."""
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("  \\end{tabular}\n")
            if self.caption:
                f.write(f"  \\caption{{{self.caption}}}\n")
            if self.label:
                f.write(f"  \\label{{{self.label}}}\n")
            f.write("\\end{table}\n")


class LatexTableDatWriter:
    """
    Class to create and append LaTeX table code into a .dat file with arbitrary columns.
    """
    def __init__(self, filepath, column_headers, caption=None, label=None):
        if not filepath.endswith('.dat'):
            raise ValueError("filepath must have a .dat extension.")
        if not column_headers or not isinstance(column_headers, (list, tuple)):
            raise ValueError("column_headers must be a non-empty list or tuple of header names.")
        self.filepath = filepath
        self.headers = column_headers
        self.caption = caption
        self.label = label
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        # Write header if new
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write("\\begin{table}[ht]\n")
                f.write("  \\centering\n")
                col_fmt = ' | '.join(['l'] * len(self.headers))
                f.write(f"  \\begin{{tabular}}{{| {col_fmt} |}}\n")
                f.write("    \\hline\n")
                f.write("    " + " & ".join(self.headers) + " \\\\ \n")
                f.write("    \\hline\n")

    def append_row(self, row):
        if len(row) != len(self.headers):
            raise ValueError(f"Row has {len(row)} entries but expected {len(self.headers)}.")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("    " + " & ".join(map(str, row)) + " \\\\ \n")
            f.write("    \\hline\n")

    def finalize(self):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("  \\end{tabular}\n")
            if self.caption:
                f.write(f"  \\caption{{{self.caption}}}\n")
            if self.label:
                f.write(f"  \\label{{{self.label}}}\n")
            f.write("\\end{table}\n")


# Example usage: compute and append to both tables
if __name__ == '__main__':
    headers = ['X1', 'X2', 'X3', 'X4', 'X5']  # adjust list size as needed
    tex_file = 'results_table.tex'
    dat_file = 'results_table.dat'

    tex_writer = LatexTableWriter(tex_file, headers, caption='Computed Xs', label='tab:xs')
    dat_writer = LatexTableDatWriter(dat_file, headers, caption='Computed Xs Dat', label='tab:xs_dat')

    for i in range(1, 11):  # replace with real iteration logic
        row = [i * j for j in range(1, len(headers)+1)]
        tex_writer.append_row(row)
        dat_writer.append_row(row)

    tex_writer.finalize()
    dat_writer.finalize()
    print(f"Wrote tables: {tex_file}, {dat_file}")
