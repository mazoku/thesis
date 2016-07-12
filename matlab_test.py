import matlab.engine

eng = matlab.engine.start_matlab()

# perim, area = eng.square_stats(2, nargout=2)
# print 'returned perimeter = %.1f\nreturned area = %.1f' % (perim, area)

eng.localized_seg_demo(nargout=0)

eng.quit()