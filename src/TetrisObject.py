# -*- coding: utf-8 -*-
import copy
import random

class PutPositionState:
    def __init__( self ):
        self._validPut = False
        self._putPos = ()

    def setValidity( self, validity ):
        self._validPut = validity

    def isValid( self ):
        return self._validPut

    def setPos( self, pos ): # the pos must be a tuple
        self._putPos = pos

    def getPos( self ):
        return self._putPos

    def getBottomRow( self ):
        return max( [ row in ( row, col ) in self._putPos ] )

class BlockMovement:
    def __init__( self, tetrisBlock ):
        self._tetrisBlock = tetrisBlock
        self._hDelta = 0         # horizontal delta( range {0,N} ) from the origin
        self._rotationCount = 0  # clockwise
        self._putPos = ()

    def setHorizontalDelta( self, delta ):
        self._hDelta = delta

    def setRotationCount( self, count ):
        self._rotationCount = count

    def setPutPos( self, pos ):
        self._putPos = pos

    def getPutPos( self ):
        return self._putPos

class TetrisContainer:
    def __init__( self, area = None ):
        if area == None:
            # 0 means empty, 1 means filled
            self._area = [ [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0] ]
        else:
            self._area = area
        self.initState()

    def getCopyArea( self ):
        return copy.deepcopy( self._area )

    def getCopyContainer( self ):
        return copy.deepcopy( self )

    def putBlockInContainer( self, fixBlock ):
        # fixBlock has 4 point and can not move to any where
        def clearFilledLine( self ):
            filledLineCount = 0
            unfilledArea = []
            for row in self._area:
                if sum( row ) == self.getColumnCount():
                    filledLineCount += 1
                else:
                    unfilledArea += [ row ]

            newArea = []
            for i in range( filledLineCount ):
                newArea += [ [0] * self.getColumnCount() ]
            newArea += unfilledArea
            self._area = newArea

            # update dynamic state
            self.lastLineClearCount = filledLineCount
            if filledLineCount != 0 :
                self.combo += 1
            else:
                self.combo = 0

        ### function statements
        self.totalBlockCount += 1        # update dynamic state
        for ( row, col ) in fixBlock:
            self._area[row][col] = 1
        clearFilledLine( self )
        self.computeStaticState()

    def getStateOfFallingBlock( self, fallingBlock ):
        # fallingBlock has 4 point and can not move but falling down
        rtnState = PutPositionState()

        # check column in the border
        for ( row, col ) in fallingBlock:
            if col >= self.getColumnCount():
                rtnState.setValidity( False )
                return rtnState

        # handle falling process
        for fallDelta in range( self.getRowCount() + 1):  # plus 1 so that row will overflow
            putPos = [ ( row + fallDelta, col ) for ( row, col ) in fallingBlock ]
            for ( row, col ) in putPos:
                if row == self.getRowCount() or self._area[row][col] == 1:
                    return rtnState
            else:
                rtnState.setValidity( True )
                rtnState.setPos( putPos )

    def printContainer( self ):
        for row in self._area:
            for grid in row:
                TetrisBlock.printGrid( grid )
            print()

    def getColumnCount( self ):
        return len( self._area[0] )

    def getRowCount( self ):
        return len( self._area )

    def initState( self ):
        self.initStaticState()
        self.initDynamicState()

    def initStaticState( self ):
        self.topFilledGridLine = [self.getRowCount()] * self.getColumnCount() # self.containerRowCount is the border
        self.filledGridCount = 0
        self.holeCount = 0      # the empty grid which are covered by higher filled grid
        self.blockadeCount = 0

    def initDynamicState( self ):
        self.lastLineClearCount = 0
        self.totalBlockCount = 0
        self.combo = 0

    def computeStaticState( self ):
        self.initStaticState()
        for ( rowIndex, gridInRow ) in enumerate( self._area ):
            self.updateTopFilledGridLine( rowIndex, gridInRow )
            self.gatherFilledGridCount( gridInRow )
            self.gatherHolesCount( rowIndex, gridInRow )
        self.computeBlockadeCount()

    def updateTopFilledGridLine( self, rowIndex, gridInRow ):
        for ( colIndex, grid ) in enumerate( gridInRow ):
            if self.topFilledGridLine[colIndex] == self.getRowCount() and grid == 1:
                self.topFilledGridLine[colIndex] = rowIndex

    def computeBlockadeCount( self ):
        for colIndex in range( self.getColumnCount() ):
            segCount = 0
            for rowIndex in range( self.getRowCount() ):
                if self._area[rowIndex][colIndex] == 1:
                    segCount += 1
                else:
                    self.blockadeCount += segCount
                    segCount = 0

    def gatherFilledGridCount( self, gridInRow ):
        self.filledGridCount += sum( gridInRow )

    def gatherHolesCount( self, rowIndex, gridInRow ):
        for ( colIndex, grid ) in enumerate( gridInRow ):
            if self.topFilledGridLine[colIndex] < rowIndex and grid == 0:
                self.holeCount += 1

    def printContainerState( self ):
        print("topfilledgrid = " + str( self.topFilledGridLine ))
        print("revtopfilledgrid = " + str( [( self.getRowCount() - r ) for r in self.topFilledGridLine] ))
        print("filledGridCount = " + str( self.filledGridCount ))
        print("holeCount = " + str( self.holeCount ))
        print("blockadeCount = "+ str( self.blockadeCount ))
        print("lastLineClearCount = " + str( self.lastLineClearCount ))
        print("totalBlockCount = " + str( self.totalBlockCount ))
        print("combo = " + str( self.combo ))

class TetrisBlock:
    # the orders are meaningful, the first one is initial coordinate in the ordinary tetris game,
    # the followings stands for rotating clockwise
    BLOCK_I = ( ( ( 0, 0 ), ( 0, 1 ), ( 0, 2 ), ( 0, 3 ) ),
                ( ( 0, 0 ), ( 1, 0 ), ( 2, 0 ), ( 3, 0 ) ) )
    BLOCK_S = ( ( ( 0, 1 ), ( 0, 2 ), ( 1, 0 ), ( 1, 1 ) ),
                ( ( 0, 0 ), ( 1, 0 ), ( 1, 1 ), ( 2, 1 ) ) )
    BLOCK_N = ( ( ( 0, 0 ), ( 0, 1 ), ( 1, 1 ), ( 1, 2 ) ),
                ( ( 0, 1 ), ( 1, 0 ), ( 1, 1 ), ( 2, 0 ) ) )
    BLOCK_O = ( ( ( 0, 0 ), ( 0, 1 ), ( 1, 0 ), ( 1, 1 ) ),)
    BLOCK_T = ( ( ( 0, 1 ), ( 1, 0 ), ( 1, 1 ), ( 1, 2 ) ),
                ( ( 0, 0 ), ( 1, 0 ), ( 1, 1 ), ( 2, 0 ) ),
                ( ( 0, 0 ), ( 0, 1 ), ( 0, 2 ), ( 1, 1 ) ),
                ( ( 0, 1 ), ( 1, 0 ), ( 1, 1 ), ( 2, 1 ) ) )
    BLOCK_L = ( ( ( 0, 2 ), ( 1, 0 ), ( 1, 1 ), ( 1, 2 ) ),
                ( ( 0, 0 ), ( 1, 0 ), ( 2, 0 ), ( 2, 1 ) ),
                ( ( 0, 0 ), ( 0, 1 ), ( 0, 2 ), ( 1, 0 ) ),
                ( ( 0, 0 ), ( 0, 1 ), ( 1, 1 ), ( 2, 1 ) ) )
    BLOCK_J = ( ( ( 0, 0 ), ( 1, 0 ), ( 1, 1 ), ( 1, 2 ) ),
                ( ( 0, 0 ), ( 0, 1 ), ( 1, 0 ), ( 2, 0 ) ),
                ( ( 0, 0 ), ( 0, 1 ), ( 0, 2 ), ( 1, 2 ) ),
                ( ( 0, 1 ), ( 1, 1 ), ( 2, 0 ), ( 2, 1 ) ) )

    ALL_BLOCK = [BLOCK_I, BLOCK_S, BLOCK_N, BLOCK_O, BLOCK_T, BLOCK_L, BLOCK_J]
    ALL_BLOCK_NAME = ["I", "S", "N", "O", "T", "L", "J"]
    BLOCK_GRID_COUNT = 4
    GRID_EMPTY = "o"   # as num 0
    GRID_FILLED = "x"  # as num 1

    def __init__( self, blockStr ):
        self._name = blockStr
        self._posAllDirection = TetrisBlock.ALL_BLOCK[TetrisBlock.ALL_BLOCK_NAME.index( blockStr )]

    @staticmethod
    def getRandBlock():
        randNum = random.randint( 0, len( TetrisBlock.ALL_BLOCK_NAME ) -1 )
        return TetrisBlock( TetrisBlock.ALL_BLOCK_NAME[randNum] )

    def getBlockName( self ):
        return self._name

    def getAllDirectionPos( self ):
        return self._posAllDirection

    def getDirectionCount( self ):
        return len( self._posAllDirection )

    def getPosAfterRotateClockWise( self, rotateCount ):
        return self._posAllDirection[ rotateCount % self.getDirectionCount() ]

    @staticmethod
    def getEmptyGrid4x4():
        return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    @staticmethod
    def printGrid( grid ):
        if ( grid == 1 ):
            print(TetrisBlock.GRID_FILLED, end="")
        else:
            print(TetrisBlock.GRID_EMPTY, end="")

    @staticmethod
    def printGrid4x4( grid4x4 ):
        for row in grid4x4:
            for grid in row:
                TetrisBlock.printGrid( grid )
            print()

    @staticmethod
    def printAllBlock():
        for oneBlock in TetrisBlock.ALL_BLOCK:
            for fixDirectionBlock in oneBlock:
                grid4x4 = TetrisBlock.getEmptyGrid4x4()
                for ( row, col ) in fixDirectionBlock:
                    grid4x4[row][col] = 1 #filled
                TetrisBlock.printGrid4x4( grid4x4 )
                print("-------------------")
            print("-------------------")

if __name__=='__main__':
    TetrisBlock.printAllBlock()
    test = TetrisContainer()
    test.printContainer()
    test.printContainerState()
