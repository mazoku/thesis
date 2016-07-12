function [perim, area] = square_stats(a)
    perim = 4 * a;
    area = a * a;
    
    disp(sprintf('perimeter = %.1f', perim))
    disp(sprintf('area = %.1f', area))
end