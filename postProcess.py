def postProcess(path):
	try:
		import matlab.engine
	except NameError:
		print("failed to initialize matlab.engine.")
		return -1
	
	import os
	eng = matlab.engine.start_matlab()      # start engine
	eng.cd( os.path.abspath(path) )         # cd into right dir
	eng.video_postProcess_python(nargout=0) # process
	eng.quit()										 # quit engine
	return 1
