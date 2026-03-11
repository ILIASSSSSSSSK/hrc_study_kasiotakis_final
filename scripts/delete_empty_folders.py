import os

def delete_empty_folders(directory):
	for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
		if not dirnames and not filenames:
			try:
				os.rmdir(dirpath)
			except Exception as e:
				print("Failed to delete {}: {}".format(dirpath, e))

if __name__ == "__main__":
	directory = "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Paper_Results/rl_models"

	delete_empty_folders(directory)