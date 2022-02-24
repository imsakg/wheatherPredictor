"""Headless SVG charts with escaped text and explicit sample-index axes."""
from html import escape
from .units import finite

_COLORS = ("#1769aa","#c64b13","#328145","#8359a3","#7c6846")


def line_chart(series, title="Weather series", width=800, height=360):
    if not isinstance(width,int) or not isinstance(height,int) or width < 200 or height < 180:
        raise ValueError("chart dimensions are too small")
    cleaned = {str(name):[finite(v) if v is not None else None for v in values] for name,values in series.items()}
    all_values = [v for values in cleaned.values() for v in values if v is not None]
    if not all_values:
        raise ValueError("chart requires finite samples")
    length = max(len(values) for values in cleaned.values())
    low, high = min(all_values), max(all_values)
    pad = max((high-low)*0.08,0.5)
    low, high = low-pad, high+pad
    def xy(index,value):
        return 62+(width-90)*index/max(1,length-1), height-52-(height-100)*(value-low)/(high-low)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {} {}">'.format(width,height),
           '<title>'+escape(str(title))+'</title>', '<rect width="100%" height="100%" fill="white"/>',
           '<text x="62" y="24" font-family="sans-serif" font-size="16">'+escape(str(title))+'</text>']
    for value in (low,(low+high)/2,high):
        _,y = xy(0,value)
        svg.append('<path d="M62 {0:.2f} H{1}" stroke="#ddd"/><text x="4" y="{0:.2f}" font-size="12">{2:.2f}</text>'.format(y,width-28,value))
    for i,(name,values) in enumerate(cleaned.items()):
        color = _COLORS[i%len(_COLORS)]
        segment = []
        for index,value in enumerate(values+[None]):
            if value is not None:
                x,y = xy(index,value)
                segment.append('{:.2f},{:.2f}'.format(x,y))
            elif segment:
                svg.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(color,' '.join(segment)))
                if len(segment)==1:
                    x,y=segment[0].split(',')
                    svg.append('<circle cx="{}" cy="{}" r="3" fill="{}"/>'.format(x,y,color))
                segment = []
        svg.append('<text x="{}" y="{}" fill="{}" font-size="12">{}</text>'.format(62+i*140,height-10,color,escape(name)))
    svg.append('<text x="62" y="{}" font-size="12">sample index: 0 to {}</text></svg>'.format(height-30,length-1))
    return ''.join(svg)
