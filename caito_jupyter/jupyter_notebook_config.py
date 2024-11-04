c = get_config()

# Set the hashed password
c.NotebookApp.password = 'sha1:f94baaa5a070:0b21b769a428bf91f84ddabd7e261e7b76c4cea1'

# Set the IP address to '*' to bind on all interfaces
c.NotebookApp.ip = '0.0.0.0'

# Set the port to 8099
c.NotebookApp.port = 8099

# Disable automatic browser launch
c.NotebookApp.open_browser = False

# (Optional) Disable token authentication
c.NotebookApp.token = ''

# (Optional) Require HTTPS
c.NotebookApp.allow_remote_access = True
