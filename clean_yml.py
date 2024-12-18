import yaml

# List of packages to clean
packages_to_clean = [
    "tornado", "kiwisolver", "libffi", "libstdcxx-ng", "wrapt", "libxcb", "icu", "libedit", 
    "tensorflow", "pyzmq", "hdf5", "git", "pyrsistent", "libssh2", "libxml2", "readline", 
    "freetype", "qt", "tensorflow-base", "libgfortran-ng", "libpng", "gmp", "pyqt", "jpeg", 
    "libsodium", "ncurses", "libopenblas", "krb5", "h5py", "numpy", "libgcc-ng", "zlib", 
    "libuuid", "python", "sqlite", "mistune", "xz", "libcurl", "tk", "markupsafe", "openssl", 
    "scikit-learn", "sip", "grpcio", "scipy", "zeromq", "c-ares", "matplotlib-base", "pcre", 
    "gstreamer", "fontconfig", "perl", "dbus", "glib", "ld_impl_linux-64"
]

# Input and output file names
input_file = "environment.yml"
output_file = "cleaned_environment.yml"

# Load the environment.yml file
with open(input_file, "r") as file:
    env_data = yaml.safe_load(file)

# Process dependencies
cleaned_dependencies = []
for dep in env_data["dependencies"]:
    if isinstance(dep, str):  # Only process string dependencies
        package_name = dep.split("=")[0]
        if package_name in packages_to_clean:
            cleaned_dependencies.append(package_name)  # Keep only the package name
        else:
            cleaned_dependencies.append(dep)  # Leave other dependencies untouched
    else:
        cleaned_dependencies.append(dep)  # Handle nested dependencies as is

# Update the dependencies in the environment data
env_data["dependencies"] = cleaned_dependencies

# Save the cleaned environment.yml file
with open(output_file, "w") as file:
    yaml.dump(env_data, file, default_flow_style=False)

print(f"Cleaned environment.yml file saved as {output_file}")
