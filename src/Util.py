def getGap( intVector ):
    return [ abs( intVector[i] - intVector[i - 1] ) for i in range( 1, len( intVector ) ) ]

def getDeviation( intVector ):
    avg = sum( intVector) / len( intVector )
    deviation = sum( ( aInt - avg )**2 for aInt in intVector )
    return deviation
