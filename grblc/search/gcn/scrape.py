import subprocess

def scrape():
    # Run a Julia script
    subprocess.run(["julia", "scrape.jl"])

    # Run Julia code directly
    result = subprocess.run(["julia", "-e", "println(1 + 2)"], capture_output=True, text=True)
    return result.stdout
