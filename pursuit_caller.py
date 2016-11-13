import glob
import multiprocessing
import time
import argparse
import FlyPursuit
import functools
# TODO:
#  - log te file: http://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
#     - monitor global progress e.g. "x/y jobs finished"
if __name__ == '__main__':
   parser = argparse.ArgumentParser( description="Wrapper for FlyPursuit. Processes all *.avi matching [--PATH].")
   parser.add_argument('-p', '--path', type=str, default="D:/jan.playback/**/*.avi", help='folder to process')
   parser.add_argument('-w', '--wait', type=int, default=None, help='wait for N minutes before starting')
   parser.add_argument('-s', '--start-time', type=int, default=None, help='start at clock time - not implemented')      
   parser.add_argument('-d', '--duration', type=int, default=None, help='maximum duration of the jobs in seconds after which jobs will be killed')      
   parser.add_argument('--processes', type=int, default=multiprocessing.cpu_count(), help='number of parallel processes to run (bounded by CPU count')
   args = parser.parse_args()
   
   multiprocessing.freeze_support() # this is required on windows
   
   if args.wait is not None: # wait N seconds
      print("waiting {0} minutes.".format(args.wait))
      time.sleep(60*args.wait) # wait N minutes

   file_names = glob.glob( args.path ) # reverse to start processing newest files
   
   pool = multiprocessing.Pool(processes=args.processes) # init worker pool
   pool.map_async(                                       # "submit" jobs to queue
         functools.partial(FlyPursuit.run, override=False), 
         file_names[::-1], chunksize=1) 
   
   if args.duration is not None: # wait for specified amount of time and terminate processes
      time.sleep(args.duration)
      pool.terminate()
      pool.join() 
      print("process terminated after {0} seconds.".format(args.duration))
   else: # close the pool and wait for the work to finish 
      pool.close() 
      pool.join() 
      print("process finished.")
      
