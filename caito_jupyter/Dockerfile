# Use the base Jupyter Notebook image
FROM jupyter/base-notebook

# Copy the custom Jupyter Notebook configuration file
COPY jupyter_notebook_config.py /home/jovyan/.jupyter/

# Expose port 8099
EXPOSE 8099

# Start Jupyter Notebook using the custom configuration
CMD ["start-notebook.sh"]
