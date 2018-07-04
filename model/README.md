- Install requirements: pip install -r requirements.txt
- View config options: python reddit.py print_config
- Train a model (and save weights to tmp/)
   1) with default options: python reddit.py
   2) with custom options: python reddit.py with max_length=50 max_posts=100
- Test a model: python test.py <path/to/weight/file>
