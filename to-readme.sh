
rm -rf SGNN-demo_files/;
rm README.md

jupyter nbconvert --to markdown SGNN-demo.ipynb

mv SGNN-demo.md README.md

