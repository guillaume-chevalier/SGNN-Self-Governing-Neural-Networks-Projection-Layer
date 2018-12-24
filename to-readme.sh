
rm -rf SGNN-demo_files/;

jupyter nbconvert --to markdown SGNN-demo.ipynb

mv SGNN-demo.md README.md
